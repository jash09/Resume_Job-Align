import PyPDF2
import google.generativeai as genai
import pandas as pd
import os
import json
import re
import functions as fn
import traceback

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
                # Attempt to generate the resume and clean the text
                response = fn.generate_resume(resume_json, job_description)
                responses.append(fn.clean_text(response))
                print(f"Done for resume {resume_index} and job {job_title}.")
                break  # If successful, break out of the retry loop
            except Exception as e:
                # print(f"Error processing resume {resume_index} for job {job_title}: {e}")
                attempts += 1
                if attempts == MAX_RETRIES:
                    print(f"Max retries reached for resume {resume_index} and job {job_title}. Skipping.")
                    responses.append(None)  # Append None or some placeholder to maintain the list's integrity
                    break

    df_resumes[job_title] = responses  # Assign responses to the DataFrame

df_resumes.to_csv('updated_resumes.csv', index=False)