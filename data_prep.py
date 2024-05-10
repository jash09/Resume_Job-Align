import pandas as pd
import functions as fn

resume_folder = '/Users/parthvinm/Desktop/NLP SSM/Final Project/Resume_Job-Align/Resumes'

df_resumes = fn.read_resumes(resume_folder)
df_resumes['resume_json'] = df_resumes.apply(fn.generate_response, axis=1)

df_jobs = pd.read_csv('/Users/parthvinm/Desktop/NLP SSM/Final Project/Resume_Job-Align/jobs.csv', encoding='iso-8859-1')
df_jobs['job_json'] = df_jobs.apply(fn.generate_response, resume=False, axis=1)

# Create a new DataFrame that merges df_resumes and df_jobs by creating a Cartesian product
df_combined = df_resumes.assign(key=1).merge(df_jobs.assign(key=1), on='key').drop('key', axis=1)
df_combined.to_csv('resume_job_pairs.csv', index=False)