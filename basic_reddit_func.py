import os
from dotenv import load_dotenv
import praw
from praw.models import MoreComments
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import time
import functools
from textblob import TextBlob
from tkinter import *
from tkinter import ttk
from tkinter import messagebox
load_dotenv() # Load environment variables from .env file

nltk.download('vader_lexicon')

# Simple timing decorator
def timer(func):
    """Basic timer decorator that prints execution time"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"{func.__name__} took {execution_time:.4f} seconds to execute")
        return result
    return wrapper

class RedditAPI:
    def __init__(self, reddit_client_id, reddit_client_secret, reddit_user_agent):
        if not all([reddit_client_id, reddit_client_secret, reddit_user_agent]):
            raise ValueError("Missing Reddit API credentials. Please check your .env file.")
        
        # Initialize Reddit API client with error handling
        try:
            self.reddit = praw.Reddit(
                client_id=reddit_client_id,
                client_secret=reddit_client_secret,
                user_agent=reddit_user_agent
            )
            
            # Try to access Reddit to verify credentials
            try:
                subreddit = self.reddit.subreddit("test")
                _ = subreddit.display_name
                print("✓ Successfully connected to Reddit API")
            except Exception as e:
                print(f"✗ Failed to connect to Reddit API: {str(e)}")
                raise
                
        except Exception as e:
            print(f"Error initializing Reddit client: {str(e)}")
            raise

def vaderAnalysis(sentence):
    sid_obj = SentimentIntensityAnalyzer()
    sentiment_dict = sid_obj.polarity_scores(sentence)
    
    if sentiment_dict['compound'] >= 0.05:
        return "Positive"
    elif sentiment_dict['compound'] <= -0.05:
        return "Negative"
    else:
        return "Neutral"

@timer
def printVaderAnalysis(entry):
    count = 0
    positive_tot = 0
    negative_tot = 0
    neutral_tot = 0
    entry.comments.replace_more(limit=None)
    for comment in entry.comments.list():
        count += 1
        formatted_comment = comment.body.replace("\n", "")
        sentiment = vaderAnalysis(formatted_comment)
        if (sentiment == "Positive"):
            positive_tot += 1
        elif (sentiment == "Negative"):
            negative_tot += 1
        elif (sentiment == "Neutral"):
            negative_tot += 1

    print("\n VADER Totals")
    print ("-------------------------------------")
    print(f"Positive = {positive_tot / count* 100}%")
    print(f"Negative = {negative_tot / count * 100}%")
    print(f"Neutral = {neutral_tot / count * 100}%")

    return positive_tot, negative_tot, neutral_tot

def analyze_sentiment_bert(sentence, model_name="cardiffnlp/twitter-roberta-base-sentiment-latest"):
    try:
        # Create sentiment analysis pipeline
        sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=model_name,
            tokenizer=model_name,
            top_k=None
        )
        
        # Perform sentiment analysis
        results = sentiment_pipeline(sentence)
        
        # Extract the results
        sentiment_scores = results[0]
        
        # Find the sentiment with highest confidence
        best_sentiment = max(sentiment_scores, key=lambda x: x['score'])
        
        # Format results
        formatted_results = {
            'sentence': sentence,
            'sentiment': best_sentiment['label'],
            'confidence': round(best_sentiment['score'], 4),
            'all_scores': {item['label']: round(item['score'], 4) for item in sentiment_scores}
        }
        
        return formatted_results
        
    except Exception as e:
        return {
            'sentence': sentence,
            'error': str(e),
            'sentiment': None,
            'confidence': None
        }
    
@timer
def printBERTAnalysis(entry):
    count = 0
    neutral_tot = 0
    positive_tot = 0
    negative_tot = 0
    entry.comments.replace_more(limit=None)
    for comment in entry.comments.list():
        time.sleep(2)
        count += 1
        formatted_comment = comment.body.replace("\n", "")
        result = analyze_sentiment_bert(formatted_comment)

        if (result['sentiment'] == 'neutral'):
            neutral_tot += 1
        elif (result['sentiment'] == 'positive'):
            positive_tot += 1
        elif (result['sentiment'] == 'negative'):
            negative_tot += 1
        
    print("\n BERT Totals")
    print ("-------------------------------------")
    print(f"Positive = {positive_tot / count* 100}%")
    print(f"Negative = {negative_tot / count * 100}%")
    print(f"Neutral = {neutral_tot / count * 100}%")

    return positive_tot, negative_tot, neutral_tot, count


def analyzeBlob(sentence):
    blob = TextBlob(sentence)
    return blob.sentiment.polarity

@timer
def printBlobAnalysis(entry):
    count = 0
    neutral_tot = 0
    positive_tot = 0
    negative_tot = 0

    entry.comments.replace_more(limit=None)
    for comment in entry.comments.list():
        count += 1
        formatted_comment = comment.body.replace("\n", "")
        result = analyzeBlob(formatted_comment)

        if (result > 0.3):
            positive_tot += 1
        elif (result < -0.3):
            negative_tot += 1
        else:
            neutral_tot += 1

    print("\n BLOB Totals")
    print ("-------------------------------------")
    print(f"Positive = {positive_tot / count* 100}%")
    print(f"Negative = {negative_tot / count * 100}%")
    print(f"Neutral = {neutral_tot / count * 100}%")

    return positive_tot, negative_tot, neutral_tot, count


if __name__ == "__main__":
    
    # Load credentials
    CLIENT_ID = os.getenv('CLIENT_ID')
    CLIENT_SECRET = os.getenv('CLIENT_SECRET')
    USER_AGENT = os.getenv('USER_AGENT')

    api_conn = RedditAPI(CLIENT_ID, CLIENT_SECRET, USER_AGENT)

    # Create the main window
    root = Tk()
    root.title("Reddit Sentiment Input Application")
    root.geometry("800x600")

    mainframe = ttk.Frame(root, padding="3 3 12 12")
    mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)
    
    ttk.Label(mainframe, text="Please enter a link").grid(column=2, row=1, sticky=W)
    # text_box = ttk.Entry(root, width=100, font=("Arial", 16))

    feet = StringVar()
    feet_entry = ttk.Entry(mainframe, width=100, textvariable=feet)
    feet_entry.grid(column=2, row=2, sticky=(W, E))
    
    # Function to handle submit button click
    def on_submit():
        user_input = feet_entry.get()
        if user_input.strip():  # Check if input is not empty
            messagebox.showinfo("Submitted", f"You entered: {user_input}")

            submission = api_conn.reddit.submission(url=user_input)

            printBlobAnalysis(submission)
            printVaderAnalysis(submission)
            printBERTAnalysis(submission)

            feet_entry.delete(0, END)  # Clear the text box
        else:
            messagebox.showwarning("Warning", "Please enter some text!")
    
    # Create submit button
    submit_button = Button(root, text="Submit", command=on_submit, 
                             font=("Arial", 12), bg="#4CAF50", fg="white")
    # submit_button.pack(pady=10)
    submit_button.grid(column=3, row=2, sticky=(W, E))

    # url = "https://www.reddit.com/r/MechanicalKeyboards/comments/1lzh921/almost_got_a_heart_attack_thinking_my_magnetic/"
    
    
    # Start the GUI event loop
    root.mainloop()
    
