import google.generativeai as genai
import pandas as pd
import functions as fn

api_key = 'AIzaSyB3E-pVTZLoxpAsj6tNhccJFaXOxVKJDB4'
genai.configure(api_key=api_key)

df_resumes = pd.read_csv('resume_job_recommendations.csv')

MAX_RETRIES = 10

for index, row in df_resumes.iterrows():
    resume_json = row['resume_json']
    job_title = row['title']
    job_description = row['job_json']
    attempts = 0
    
    while attempts < MAX_RETRIES:
        try:
            response = fn.generate_resume(resume_json, job_description)
            df_resumes.loc[index, 'new_resume'] = response
            print(f"Done for {row['file_name']} and job {job_title}.")
            break  
        except Exception as e:
            attempts += 1
            if attempts == MAX_RETRIES:
                print(f"Max retries reached for {row['file_name']} and job {job_title}. Skipping.")
                df_resumes.loc[index, 'new_resume'] = None
                break

df_resumes.to_csv('new_resumes.csv', index=False)