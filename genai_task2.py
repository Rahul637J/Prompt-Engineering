'''
@Author: Rahul
@Date: 2024-11-22
@Last Modified by: Rahul
@Last Modified time: 2024-11-22
@Title: Python program to process reviews and replyin to the reviews
'''

import google.generativeai as genai
import pandas as pd
from dotenv import load_dotenv
import os

# Function to read email content from a text file
def read_reviews_file(file_path):
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
            review_content = file.read()
        return [review.strip() for review in review_content.split("---END OF REVIEW---") if review.strip()]
    except Exception as e:
        print(f"Error reading the file: {e}")
        return []

# Function to summarize email content using the Gemini API
def guessing_product_by_review(review, chat_session):
    '''
    Description:
        Summarizes the given text using the Gemini API with a designed prompt and optional history.
    
    Parameters:
        text (str): Text to summarize.
        history (list): List of previous prompts and responses for context (optional).
    
    Return Type:
        tuple: Summary of the text and updated history.
    '''
    prompt = f"Give only the product by analysing this review: {review}"
    try:
        response = chat_session.send_message(prompt)
        return response.text
    
    except Exception as e:
        print(f"Error during summarization: {e}")
        return "Summarization failed"

# Function to translate text using the Gemini API
def analyse_the_sentiment(review, chat_session):
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
    try:
        response = chat_session.send_message(f"Try to analyse the sentiment of the {review} like 'Positive','Negative','Neutral' give only result among these 3")
        return response.text
    except Exception as e:
        print(f"Error during translation: {e}")
        return "Translation failed"

# Function to extract email details
def replying_based_sentiment(review,sentiment,chat_session):
    '''
    Description:
        Extracts sender, receiver, and email body from the given email content.
    
    Parameters:
        email (str): Email content.
    
    Return Type:
        tuple: Sender, receiver, and body of the email.
    '''
    
    response = chat_session.send_message(f'Give 1 line scentence for this review: "{review}" based on the sentiment:{sentiment} like "Thank you for feedback" and give reply for that add the thank for your feedback.')
    return response.text
    
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
        
    genai.configure(api_key=api_key)
    
    generation_config = {
        "temperature": 0.5,
        "top_p": 0.65,
        "top_k": 64,
        "max_output_tokens": 8192,
        }
    
    model = genai.GenerativeModel(model_name="gemini-1.5-flash", generation_config=generation_config)
    chat_session = model.start_chat(history=[])
    
    reviews = read_reviews_file('reviews.txt')
        
    if reviews:
        review_details = []

        for review in reviews:
            
            # Get the first item from the list
            original_name = review.split('\n')[0].split(': ')[1]
            review_content = review.split('\n')[1].split(': ')[1]
            
            guessed_product = guessing_product_by_review(review_content,chat_session)
                        
            # Summarize the email body with history
            sentiment = analyse_the_sentiment(review, chat_session)
            
            # # Translate the summary with history
            review_reply = replying_based_sentiment(review,sentiment,chat_session)
            
            review_details.append({
                "Original Product": original_name,
                "Guessed Product": guessed_product,
                "Sentiment": sentiment,
                "Review": review_content,
                "Reply": review_reply 
            })

        # Save results to a CSV
        df = pd.DataFrame(review_details)
        output_file = "Processed_Review.csv"
        df.to_csv(output_file, index=False)
        print(f"Process completed successfully. Data saved to '{output_file}'.")
    else:
        print("No emails to process.")

if __name__ == "__main__":
    main()
