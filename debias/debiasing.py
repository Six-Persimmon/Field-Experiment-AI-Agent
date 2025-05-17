#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
@File    :   debiasing.py
@Time    :   2025/05/16 22:14:32
@Author  :   Manlu Ouyang
@Version :   1.0
@Contact :   mo2615@stern.nyu.edu
@Desc    :   None
'''

import openai
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import Ridge
import os

openai.api_key = os.getenv("OPENAI_API_KEY")  

## tool functions
def get_embedding(text, model="text-embedding-3-small"):
    response = openai.embeddings.create(input=[text], model=model)
    return response.data[0].embedding

def fit_transfer_beta(df_osf, df_qual, w=0.5, lambda_ridge=1.0):
    """
    Fit transfer ridge regression using only question embeddings.
    
    Parameters:
    - df_osf: DataFrame with OSF data (must contain 'Embedding', 'Average_LLM_Response', 'Average_Human_Response')
    - df_qual: DataFrame with Qualtrics data (same structure as df_osf)
    - w: weight for source data
    - lambda_ridge: ridge regularization strength
    
    Returns:
    - beta_hat: numpy array of shape (embedding_dim,)
    """
    
    # Stack embeddings into matrix X
    X_hist = np.vstack(df_osf["Embedding"])
    X_new = np.vstack(df_qual["Embedding"])

    # Compute biases: delta = LLM - Human
    y_hist = df_osf["Average_LLM_Response"].to_numpy() - df_osf["Average_Human_Response"].to_numpy()
    y_new = df_qual["Average_LLM_Response"].to_numpy() - df_qual["Average_Human_Response"].to_numpy()

    # Combine design matrix and response
    X_combined = np.vstack([X_hist, X_new])
    y_combined = np.concatenate([w * y_hist, (1 - w) * y_new])

    # Sample weights (optional — for weighted regression)
    sample_weights = np.concatenate([
        np.full(len(y_hist), w),
        np.full(len(y_new), 1 - w)
    ])

    # Fit ridge regression without intercept
    model = Ridge(alpha=lambda_ridge, fit_intercept=False)
    model.fit(X_combined, y_combined, sample_weight=sample_weights)

    return model.coef_


def get_debiased_response(df_osf_, df_qual_, df_new_, w_=0.5, lambda_ridge_=1.0):
    """
    - df_osf_: DataFrame with OSF data (must contain "Question", 'Average_LLM_Response', 'Average_Human_Response')
    - df_qual_: DataFrame with Qualtrics data (same structure as df_osf)
    - df_new_: DataFrame with new questions and LLM repsonses (must contain "Question", 'LLM_Response')
    """
    # Add embedding
    tqdm.pandas()
    df_osf_["Embedding"] = df_osf_["Question"].progress_apply(get_embedding)
    df_qual_["Embedding"] = df_qual_["Question"].progress_apply(get_embedding)

    # Save to new CSVs with embeddings
    # Use pickle to keep list structure
    df_osf_.to_pickle("hist_with_embeddings.pkl") 
    df_qual_.to_pickle("new_with_embeddings.pkl")
    print('Save CSVs with embeddings')

    beta_hat = fit_transfer_beta(df_osf_, df_qual_, w=w_, lambda_ridge=lambda_ridge_)
    print(beta_hat.shape)  

    # STEP 1 — Get embeddings for each unique question
    unique_questions = df_new_["Question"].unique()
    # Map question → embedding
    question_to_embedding = {
        q: get_embedding(q) for q in unique_questions
        }

    debiased_responses = []
    for idx, row in df_new_.iterrows():
        question = row["Question"]
        llm_response = row["LLM_Response"]

        embedding = np.array(question_to_embedding[question])
        predicted_bias = embedding @ beta_hat
        debiased = llm_response - predicted_bias

        debiased_responses.append(debiased)

    df_new_["Debiased_Response"] = debiased_responses
    df_new_.to_csv("new_question_debiased.csv", index=False)

    return beta_hat, df_osf_, df_qual_, df_new_


#### Example
# Load OSF and Qualtrics data
## Deal with missing data
# df_osf = pd.read_csv("fake_survey_data.csv")
# df_qual = pd.read_csv("new_fake_survey.csv")
# df_new = pd.read_csv("new_question.csv")
# beta_hat, df_osf_, df_qual_, df_new_ = get_debiased_response(df_osf, df_qual, df_new, w_=0.5, lambda_ridge_=1.0)