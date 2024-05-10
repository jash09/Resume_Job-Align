import google.generativeai as genai
import pandas as pd
import functions as fn

import os
os.environ["OPENAI_API_KEY"] = "sk-proj-Cw1drZV2K9ag9wXRLR2pT3BlbkFJZ9SiI26TtRgTVc9dT1UE" 

from openai import OpenAI
client = OpenAI()

df_combined = pd.read_csv('resume_job_pairs_gpt.csv')

MAX_RETRIES = 10

for index, row in df_combined.iterrows():
    resume_json = row['resume_json']
    job_title = row['title']
    job_description = row['job_json']

    attempts = 0
    while attempts < MAX_RETRIES:
        try:
            response = fn.generate_rec_gpt(resume_json, job_description)
            df_combined.loc[index, 'recommendation'] = response
            print(f"Done for {row['file_name']} and job {job_title}.")
            break  
        except Exception as e:
            attempts += 1
            if attempts >= MAX_RETRIES:
                print(f"Max retries reached for {row['file_name']} and job {job_title}. Skipping.")
                df_combined.loc[index, 'recommendation'] = None
                break  

df_combined.to_csv('resume_job_recs_gpt.csv', index=False)
