#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   simulate_response.py
@Time    :   2025/05/15 15:32:45
@Author  :   Shijian Liu
@Version :   1.0
@Contact :   lshijian405@gmail.com
@Desc    :   None

'''

import os
import yaml
import json
import warnings
import asyncio
import tempfile
import shutil
import copy
import logging
from datetime import datetime
from typing import Literal, Dict, List, Any, Union, Optional
import io
import time
import pandas as pd
import logging
import re
import subprocess
from llm_openai import openai_llm
from tqdm import tqdm



def run_single_survey_response(llm, survey_prompt_template, survey_context, participant_info):
    """
    Generate a single survey response using LLM.

    Args:
        llm: callable that takes in a prompt and returns a response.
        survey_prompt_template: str with placeholders like $age, $gender, $race.
        survey_context: json string with survey context. Contains questions and instructions.
        participant_info: dict with keys "name", "age", "gender", "race".

    Returns:
        dict with participant info and response text.
    """
    background_prompt = survey_prompt_template.replace("$age", str(participant_info["age"])) \
                                          .replace("$gender", participant_info["gender"]) \
                                          .replace("$race", participant_info["race"])
    survey_context = json.loads(survey_context)
    # combine the background prompt with the survey context to let the LLM response
    questions = survey_context["questions"]
    prompt_body = "\n".join([
        f"Q{i+1}: {q['question_text']}\nOptions: {', '.join(q['input_config']['options'])}"
        for i, q in enumerate(questions)
    ])
    full_prompt = f"{background_prompt}\n\nSurvey Theme: {survey_context['theme']}\nPurpose: {survey_context['purpose']}\n\nPlease answer the following questions in JSON format:\n\n{prompt_body}"


    response = llm(full_prompt)
    return {
        "name": participant_info["name"],
        "age": participant_info["age"],
        "gender": participant_info["gender"],
        "race": participant_info["race"],
        "response": response
    }


def run_all_survey_responses(llm, participant_csv_path, survey_prompt_template, survey_context):
    """
    Run the survey across all participants listed in the CSV.

    Args:
        llm: callable that returns LLM response.
        participant_csv_path: path to participant CSV file.
        survey_prompt_template: string with placeholders.

    Returns:
        pd.DataFrame with all responses.
    """
    df_participants = pd.read_csv(participant_csv_path)
    responses = []

    for _, row in tqdm(df_participants.iterrows(), total=len(df_participants)):
        participant_info = {
            "name": row["Name"],
            "age": row["Age"],
            "gender": row["Gender"],
            "race": row["Race"]
        }
        response_record = run_single_survey_response(
            llm, survey_prompt_template, survey_context, participant_info
        )
        responses.append(response_record)

    return pd.DataFrame(responses)