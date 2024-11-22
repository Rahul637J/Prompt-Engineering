'''
@Author: Rahul
@Date: 2024-11-21
@Last Modified by: Rahul
@Last Modified time: 2024-11-21
@Title: Python program to process emails and summarize with Gemini API
'''

import google.generativeai as genai
import pandas as pd
from dotenv import load_dotenv
import os

# Function to read email content from a text file
def read_emails(file_path):
    '''
    Description:
        Reads the email content from a file and splits it into individual emails.
    
    Parameters:
        file_path (str): Path to the file containing the email data.
    
    Return Type:
        list: A list of email strings.
    '''
    try:
        with open(file_path, "r") as file:
            email_content = file.read()
        return [email.strip() for email in email_content.split("---END OF EMAIL---") if email.strip()]
    except Exception as e:
        print(f"Error reading the file: {e}")
        return []

# Function to summarize email content using the Gemini API
def summarize_with_gemini(text, chat_session):
    '''
    Description:
        Summarizes the given text using the Gemini API with a designed prompt and optional history.
    
    Parameters:
        text (str): Text to summarize.
        history (list): List of previous prompts and responses for context (optional).
    
    Return Type:
        tuple: Summary of the text and updated history.
    '''
    prompt = f"Summarize the following email content:\n{text}"
    try:
        response = chat_session.send_message(prompt)
        return response.text
    
    except Exception as e:
        print(f"Error during summarization: {e}")
        return "Summarization failed"

# Function to translate text using the Gemini API
def translate_with_gemini(text, chat_session):
    '''
    Description:
        Translates the given text into the specified language using the Gemini API with a designed prompt and optional history.
    
    Parameters:
        text (str): Text to translate.
        target_lang (str): Target language for translation.
        history (list): List of previous prompts and responses for context (optional).
    
    Return Type:
        tuple: Translated text and updated history.
    '''
    prompt = f"Translate this email summary to only tamil :\n{text}"
    try:
        response = chat_session.send_message(prompt)
        return response.text
    except Exception as e:
        print(f"Error during translation: {e}")
        return "Translation failed"

# Function to extract email details
def extract_email_details(email):
    '''
    Description:
        Extracts sender, receiver, and email body from the given email content.
    
    Parameters:
        email (str): Email content.
    
    Return Type:
        tuple: Sender, receiver, and body of the email.
    '''
    lines = email.split("\n")
    sender, receiver, body = "", "", ""

    for line in lines:
        if line.lower().startswith("from:"):
            sender = line.split(":", 1)[1].strip()
        elif line.lower().startswith("to:"):
            receiver = line.split(":", 1)[1].strip()
        elif not line.lower().startswith(("subject:", "from:", "to:")):
            body += line.strip() + " "

    return sender, receiver, body.strip()



# Main function
def main():
    
    # Load the environment variables from the .env file
    load_dotenv()

    # Retrieve the API key from the environment
    api_key = os.getenv("GEMINI_API_KEY")

    # Configure the Gemini API with the loaded API key
    if api_key:
        genai.configure(api_key=api_key)
    else:
        print("API key not found. Make sure the .env file is configured correctly.")
        exit()
    
    emails = read_emails('emails.txt')
    
    genai.configure(api_key=api_key)
    
    generation_config = {
        "temperature": 0.3,
        "top_p": 0.50,
        "top_k": 64,
        "max_output_tokens": 512,
        }
    
    model = genai.GenerativeModel(model_name="gemini-1.5-flash", generation_config=generation_config)
    chat_session = model.start_chat(history=[])
    
    if emails:
        email_details = []

        for email in emails:
            sender, receiver, body = extract_email_details(email)
            
            # Summarize the email body with history
            summary = summarize_with_gemini(body, chat_session)

            # Translate the summary with history
            translated_summary = translate_with_gemini(summary, chat_session)

            email_details.append({
                "FROM": sender,
                "TO": receiver,
                "SUMMARY": summary,
                "SUMMARY(Translated)": translated_summary
            })

        # Save results to a CSV
        df = pd.DataFrame(email_details)
        output_file = "Processed_email_summaries_with_context.csv"
        df.to_csv(output_file, index=False)
        print(f"Process completed successfully. Data saved to '{output_file}'.")
    else:
        print("No emails to process.")

if __name__ == "__main__":
    main()
