import PyPDF2
import google.generativeai as genai
import pandas as pd
import os
import json
import re
import functions as fn

api_key = 'AIzaSyB3E-pVTZLoxpAsj6tNhccJFaXOxVKJDB4'
genai.configure(api_key=api_key)

resume_folder = '/Users/parthvinm/Desktop/NLP SSM/Final Project/Resume_Job-Align/Resumes'

df_resumes = fn.read_resumes(resume_folder)
df_resumes['resume_json'] = df_resumes.apply(fn.generate_response, axis=1)

df_jobs = pd.read_csv('/Users/parthvinm/Desktop/NLP SSM/Final Project/Resume_Job-Align/jobs.csv', encoding='iso-8859-1')
df_jobs['job_json'] = df_jobs.apply(fn.generate_response, resume = False, axis=1)

MAX_RETRIES = 10

for job_index, job in df_jobs.iterrows():
    job_title = job['title']
    job_description = job['job_json']

    responses = []  # Initialize an empty list for the responses

    for resume_index, resume in df_resumes.iterrows():
        resume_json = resume['resume_json']
        attempts = 0
        
        while attempts < MAX_RETRIES:
            try:
                # Attempt to generate the recommendation and append it
                response = fn.generate_rec(resume_json, job_description)
                responses.append(response)
                print(f"Done for resume {resume_index} and job {job_title}.")
                break  # Success, break out of the retry loop
            except Exception as e:
                attempts += 1
                if attempts >= MAX_RETRIES:
                    # Log error and append a placeholder indicating failure
                    print(f"Max retries reached for resume {resume_index} and job {job_title}. Skipping.")
                    responses.append(None)
                    break  # Skip to the next resume after max retries

    df_resumes[job_title] = responses

df_resumes.to_csv('resume_update_recs.csv', index=False)