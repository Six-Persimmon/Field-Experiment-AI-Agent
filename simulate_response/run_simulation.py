#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   run_simulation.py
@Time    :   2025/05/15 16:03:28
@Author  :   Shijian Liu
@Version :   1.0
@Contact :   lshijian405@gmail.com
@Desc    :   None
'''
from simulate_response import run_all_survey_responses
from llm_openai import openai_llm

with open("survey_prompt_template.txt", "r") as f:
    survey_template = f.read()

responses_df = run_all_survey_responses(
    llm=openai_llm,
    participant_csv_path="participants.csv",
    survey_prompt_template=survey_template
)

responses_df.to_csv("simulated_survey_responses.csv", index=False)
print("Saved responses to simulated_survey_responses.csv")