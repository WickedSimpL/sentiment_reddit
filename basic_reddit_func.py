import os
from dotenv import load_dotenv
import praw
from praw.models import MoreComments
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
load_dotenv() # Load environment variables from .env file

nltk.download('vader_lexicon')

class RedditAPI:
    def __init__(self, reddit_client_id, reddit_client_secret, reddit_user_agent):
        """
        Initialize the Reddit API client and BERT model for sentiment analysis
        """
        # Debug: Print credential status (without revealing actual values)
        print("Checking Reddit credentials...")
        print(f"CLIENT_ID provided: {'Yes' if reddit_client_id else 'No'}")
        print(f"CLIENT_SECRET provided: {'Yes' if reddit_client_secret else 'No'}")
        print(f"USER_AGENT provided: {'Yes' if reddit_user_agent else 'No'}")
        
        if not all([reddit_client_id, reddit_client_secret, reddit_user_agent]):
            raise ValueError("Missing Reddit API credentials. Please check your .env file.")
        
        # Initialize Reddit API client with error handling
        try:
            self.reddit = praw.Reddit(
                client_id=reddit_client_id,
                client_secret=reddit_client_secret,
                user_agent=reddit_user_agent
            )
            
            print("Testing Reddit connection...")
            print(f"Read-only mode: {self.reddit.read_only}")
            
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
    # print(f"Sentence = {sentence}")
    sid_obj = SentimentIntensityAnalyzer()
    sentiment_dict = sid_obj.polarity_scores(sentence)

    print(f"Sentiment Scores: {sentiment_dict}")
    print(f"Negative Sentiment: {sentiment_dict['neg']*100}%")
    print(f"Neutral Sentiment: {sentiment_dict['neu']*100}%")
    print(f"Positive Sentiment: {sentiment_dict['pos']*100}%")
    print(f"Compound Sentiment: {sentiment_dict['compound']*100}%")
    
    if sentiment_dict['compound'] >= 0.05:
        print("Overall Sentiment: Positive")
        return "Positive"
    elif sentiment_dict['compound'] <= -0.05:
        print("Overall Sentiment: Negative")
        return "Negative"
    else:
        print("Overall Sentiment: Neutral")
        return "Neutral"
        
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
        print(f"{count} : {formatted_comment} \n")

    print("\n VADER Totals")
    print ("-------------------------------------")
    print(f"Positive = {positive_tot / count* 100}%")
    print(f"Negative = {negative_tot / count * 100}%")
    print(f"Neutral = {neutral_tot / count * 100}%")

def analyze_sentiment_bert(sentence, model_name="cardiffnlp/twitter-roberta-base-sentiment-latest"):
    try:
        # Create sentiment analysis pipeline
        sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=model_name,
            tokenizer=model_name,
            return_all_scores=True
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
def printBERTAnalysis(entry):

    entry.comments.replace_more(limit=None)
    for comment in entry.comments.list():
        formatted_comment = comment.body.replace("\n", "")
        print("=== RoBERTa Sentiment Analysis ===")
        result = analyze_sentiment_bert(formatted_comment)
        print(f"Sentence: {result['sentence']}")
        print(f"Sentiment: {result['sentiment']} (Confidence: {result['confidence']})")
        print(f"All scores: {result['all_scores']}")
        print("-" * 50)


if __name__ == "__main__":
    
    # Load credentials
    CLIENT_ID = os.getenv('CLIENT_ID')
    CLIENT_SECRET = os.getenv('CLIENT_SECRET')
    USER_AGENT = os.getenv('USER_AGENT')

    api_conn = RedditAPI(CLIENT_ID, CLIENT_SECRET, USER_AGENT)

    url = "https://www.reddit.com/r/todayilearned/comments/1lzj1ns/til_prince_charles_princess_diana_only_met_in/"
    submission = api_conn.reddit.submission(url=url)

    printVaderAnalysis(submission)
    #printBERTAnalysis(submission)
