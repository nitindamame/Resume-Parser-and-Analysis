import streamlit as st
import os
import pypdf
from PIL import Image  
from dotenv import load_dotenv
from Models import cosine
import pandas as pd
import json

load_dotenv()  ## load all our environment variables

# Import the Generative AI model if needed
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

response_schema ="""{
        "JDMatch": "80%",
        "MissingKeywords": ""
}"""


def get_gemini_response(input):
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(input)
    return response.text


def input_pdf_text(uploaded_files):
    
    reader = pypdf.PdfReader(uploaded_files)
    text = ""
    for page in range(len(reader.pages)):
        page = reader.pages[page]
        text += str(page.extract_text())
    return text

@st.cache_data
def get_key_info(text):
  """
  Extracts key information from the resume using Gemini
  """
  key_info_prompt = """
  This is a resume for a job applicant. Please analyze the text and provide the following information in a clear and concise format:

  * Name
  * Contact Information (Phone, Email, LinkedIn, Github, etc)
  * Education (Degree, Year, Institution, marks)
  * Work Experience/Internships (Company, Dates, Description)
  * Skills
  * Relevant Coursework

  Resume: {text}
  """
  response = get_gemini_response(key_info_prompt.format(text='\n'.join(text)))
  return response


# Prompt Template
input_prompt = """
Hey Act Like a skilled or very experience ATS(Application Tracking System)
with a deep understanding of tech field,software engineering. Your task is to evaluate the resume based on the given job description.
You must consider the job market is very competitive. Assign the percentage Matching based 
on Job description and
the missing keywords with high accuracy
resume:{text}
description:{JD}

##Rules:
- The resume should be evaluated based on the job description.
- Missingkeywords should be based on job descriptions of skills and experience and should be one string separated by commas.
- Output should be in JSON format with the following schema {response_schema} without any deviation.

##Output :
I want the below response in JSON format with as following {response_schema}
"""

def compare(resume_texts, JD_text,response_schema, embedding_method='Gemini'):
    if embedding_method == 'Gemini':
        response = get_gemini_response(input_prompt.format(text='\n'.join(resume_texts), JD=JD_text,response_schema=response_schema))
        return response
    else:
        return "Invalid embedding method selected."

## streamlit app
st.title("Resume Parsing and Analysis ")


# Define uploaded_file outside the tab selection
uploaded_files = st.file_uploader(
    'Choose your resume.pdf file: ', type="pdf", help="Please upload the pdf", accept_multiple_files=True
)

# Tab selection
tab_selection = st.radio("Select Functionality", ["Extract key information", "Compare with Job description"])

if tab_selection == "Extract key information":
    data =[]
  
    if uploaded_files:
        srno = 1
        for uploaded_file in uploaded_files:
            print(uploaded_file.name)
            text = input_pdf_text(uploaded_file)
            key_info = get_key_info(text)
            data.append([srno,uploaded_file.name, key_info] )
            srno+=1
        df = pd.DataFrame(data, columns=["Sr.no", "File Name", "key_info"])  
        st.write(df)  
    else:
        st.subheader("Please upload a resume !!")
    # Create a DataFrame from the provided data

    # Display the table in Streamlit     
    # Display the table in Streamlit
     
        
elif tab_selection == "Compare with Job description":
    if uploaded_files:
        JD = st.text_area("**Enter the job description:**")
        embedding_method = st.selectbox("Select Embedding Method", ['Gemini', 'HuggingFace-BERT', 'Doc2Vec'])

        submit = st.button("Submit")

        if submit:
            srno = 1
            data =[]
            for uploaded_file in uploaded_files:
                response = ""
                text = input_pdf_text(uploaded_file)
                response = compare([text], JD,response_schema, embedding_method)
                print(response)
                # Convert the string to a dictionary
                data_dict = json.loads(response)
                # Access the values
                jd_match = data_dict.get("JDMatch")
                missing_keywords = data_dict.get("MissingKeywords")
                data.append([uploaded_file.name, jd_match,missing_keywords ] )
                srno+=1
            df = pd.DataFrame(data, columns=["File Name", "Match %", "Missing Keywords"])  
            st.write(df)  


    else:
        st.subheader("Please upload a resume !!")
