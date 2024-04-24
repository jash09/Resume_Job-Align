import PyPDF2
import google.generativeai as genai
import pandas as pd
import os
import json
import re

api_key = 'AIzaSyB3E-pVTZLoxpAsj6tNhccJFaXOxVKJDB4'
genai.configure(api_key=api_key)

df_prompts = pd.read_csv('prompts.csv')

def input_pdf_setup(file_path):
    text = ''
    if file_path is not None:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text()
    return text

def clean_text(output):
    # Remove "```json" from the beginning
    output = re.sub(r'^```json\n', '', output)

    # Remove text before the first '{'
    output = re.sub(r'^[^{]*', '', output)
    
    # Remove text after the last '}'
    output = re.sub(r'[^}]*$', '', output)
    
    # Remove "```" from the end, if it's still there
    output = re.sub(r'```$', '', output)

    return output

def clean_json(data):
    # Remove newlines and extra spaces
    cleaned_data = re.sub(r'\n\s*', '', data)
    
    # Remove extra spaces after colons and commas
    cleaned_data = re.sub(r'\s*:\s*', ': ', cleaned_data)
    cleaned_data = re.sub(r',\s*', ', ', cleaned_data)
    
    # Remove unnecessary escape characters
    cleaned_data = re.sub(r'\\', '', cleaned_data)
    
    return cleaned_data

def read_resumes(folder_path):
    # List all PDF files in the specified folder
    files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
    
    # Prepare a list to hold data
    data = []
    
    # Process each file
    for file_name in files:
        file_path = os.path.join(folder_path, file_name)
        resume_text = input_pdf_setup(file_path)
        data.append({'file_name': file_name, 'resume_text': resume_text})
    
    # Create a DataFrame
    df = pd.DataFrame(data)
    return df

model = genai.GenerativeModel('gemini-pro')

def generate_response(row, resume = True):
    if resume:
        prompt_structure = df_prompts.loc[df_prompts['prompt_name'] == 'prompt_structure', 'prompt_text'].values[0]
        response = model.generate_content([prompt_structure, row['resume_text']])
    else:
        prompt_job_description = df_prompts.loc[df_prompts['prompt_name'] == 'prompt_job_description', 'prompt_text'].values[0]
        response = model.generate_content([prompt_job_description, row['job_description']])
    return clean_text(response.text)
 
def generate_resume(resume_json, job_description):
    resume_json_object = json.loads(resume_json)
    work_experience = json.dumps(resume_json_object['work_experience'])
    projects = json.dumps(resume_json_object['projects'])
    
    work_exp_prompt = df_prompts.loc[df_prompts['prompt_name'] == 'work_exp_prompt', 'prompt_text'].values[0]
    work_exp = model.generate_content([work_exp_prompt, job_description, work_experience],
                                      generation_config={
                                               "temperature": 0.36,
                                                "max_output_tokens": 4000,
                                                "top_p": 0.95})
    
    prompt_project = df_prompts.loc[df_prompts['prompt_name'] == 'prompt_project', 'prompt_text'].values[0]
    work_exp_object = json.loads(clean_json(work_exp.text))
    projects = model.generate_content([prompt_project, job_description, projects],
                                      generation_config={
                                               "temperature": 0.35,
                                                "max_output_tokens": 4000,
                                                "top_p": 0.95})
    
    projects_object = json.loads(clean_json(projects.text))
    resume_json_object['work_experience'] = json.dumps(work_exp_object['work_experience'])
    resume_json_object['projects'] = json.dumps(projects_object['projects'])
    return json.dumps(resume_json_object)

def generate_rec(job_description, resume_json):
    improvement_prompt = df_prompts.loc[df_prompts['prompt_name'] == 'improvement_prompt', 'prompt_text'].values[0]
    response = model.generate_content([improvement_prompt, job_description, resume_json],
                                      generation_config={
                                               "temperature": 0.35,
                                                "max_output_tokens": 4000,
                                                "top_p": 0.95})
    return response.text