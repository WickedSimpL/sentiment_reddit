
import praw
import pandas as pd
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime
import re
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class SentimentAnalyzer:
    def __init__(self):
        """Initialize the sentiment analyzer and Reddit API connection"""
        # Load Reddit API credentials from environment variables
        self.CLIENT_ID = os.getenv('CLIENT_ID')
        self.CLIENT_SECRET = os.getenv('CLIENT_SECRET')
        self.USER_AGENT = os.getenv('USER_AGENT')
        
        # Initialize sentiment analyzer
        self.vader_analyzer = SentimentIntensityAnalyzer()
        self.reddit = None
        
        # Initialize Reddit connection
        self.initialize_reddit()
    
    def initialize_reddit(self):
        """Initialize Reddit API connection"""
        try:
            if not all([self.CLIENT_ID, self.CLIENT_SECRET, self.USER_AGENT]):
                raise ValueError("Reddit API credentials not found in environment variables")
                
            self.reddit = praw.Reddit(
                client_id=self.CLIENT_ID,
                client_secret=self.CLIENT_SECRET,
                user_agent=self.USER_AGENT
            )
            # Test the connection
            self.reddit.user.me()
            print("Reddit API connection successful!")
            return True
        except Exception as e:
            print(f"Failed to connect to Reddit API: {str(e)}")
            self.reddit = None
            return False
    
    def clean_text(self, text):
        """Clean and preprocess text for sentiment analysis"""
        if not text:
            return ""
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Remove user mentions and subreddit mentions
        text = re.sub(r'@\w+|u/\w+|r/\w+', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def analyze_sentiment_textblob(self, text):
        """Analyze sentiment using TextBlob"""
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        
        if polarity > 0.1:
            return 'positive', polarity
        elif polarity < -0.1:
            return 'negative', polarity
        else:
            return 'neutral', polarity
    
    def analyze_sentiment_vader(self, text):
        """Analyze sentiment using VADER"""
        scores = self.vader_analyzer.polarity_scores(text)
        compound = scores['compound']
        
        if compound >= 0.05:
            return 'positive', compound
        elif compound <= -0.05:
            return 'negative', compound
        else:
            return 'neutral', compound
    
    def get_posts(self, subreddit_name, limit=100, time_filter='week', progress_callback=None):
        """
        Fetch posts from a subreddit
        
        Args:
            subreddit_name (str): Name of the subreddit
            limit (int): Number of posts to fetch
            time_filter (str): Time filter ('hour', 'day', 'week', 'month', 'year', 'all')
            progress_callback (function): Optional callback function to report progress
        
        Returns:
            list: List of post dictionaries
        """
        if not self.reddit:
            raise Exception("Reddit API not initialized")
            
        posts = []
        
        try:
            subreddit = self.reddit.subreddit(subreddit_name)
            
            for i, post in enumerate(subreddit.top(time_filter=time_filter, limit=limit)):
                posts.append({
                    'title': post.title,
                    'text': post.selftext,
                    'score': post.score,
                    'num_comments': post.num_comments,
                    'created_utc': datetime.fromtimestamp(post.created_utc),
                    'url': post.url,
                    'id': post.id
                })
                
                # Report progress if callback provided
                if progress_callback:
                    progress = (i + 1) / limit * 50  # First 50% for fetching
                    progress_callback(progress)
                
        except Exception as e:
            raise Exception(f"Error fetching posts from r/{subreddit_name}: {str(e)}")
            
        return posts
    
    def analyze_posts(self, posts, progress_callback=None):
        """
        Analyze sentiment of posts
        
        Args:
            posts (list): List of post dictionaries
            progress_callback (function): Optional callback function to report progress
        
        Returns:
            pandas.DataFrame: DataFrame with posts and sentiment analysis
        """
        results = []
        
        for i, post in enumerate(posts):
            # Combine title and text for analysis
            combined_text = f"{post['title']} {post['text']}"
            clean_combined = self.clean_text(combined_text)
            
            if clean_combined:
                # TextBlob analysis
                tb_sentiment, tb_score = self.analyze_sentiment_textblob(clean_combined)
                
                # VADER analysis
                vader_sentiment, vader_score = self.analyze_sentiment_vader(clean_combined)
                
                results.append({
                    'post_id': post['id'],
                    'title': post['title'],
                    'text': post['text'][:200] + '...' if len(post['text']) > 200 else post['text'],
                    'score': post['score'],
                    'num_comments': post['num_comments'],
                    'created_date': post['created_utc'],
                    'textblob_sentiment': tb_sentiment,
                    'textblob_score': tb_score,
                    'vader_sentiment': vader_sentiment,
                    'vader_score': vader_score,
                    'url': post['url']
                })
            
            # Report progress if callback provided
            if progress_callback:
                progress = 50 + (i + 1) / len(posts) * 50  # Second 50%
                progress_callback(progress)
        
        return pd.DataFrame(results)
    
    def analyze_subreddit(self, subreddit_name, limit=100, time_filter='week', progress_callback=None):
        """
        Perform complete sentiment analysis on a subreddit
        
        Args:
            subreddit_name (str): Name of the subreddit
            limit (int): Number of posts to analyze
            time_filter (str): Time filter for posts
            progress_callback (function): Optional callback function to report progress
        
        Returns:
            pandas.DataFrame: DataFrame with posts and sentiment analysis
        """
        if not self.reddit:
            raise Exception("Reddit API not initialized. Check your credentials.")
        
        # Fetch posts
        posts = self.get_posts(subreddit_name, limit, time_filter, progress_callback)
        
        if not posts:
            raise Exception(f"No posts found in r/{subreddit_name}")
        
        # Analyze sentiment
        results_df = self.analyze_posts(posts, progress_callback)
        
        return results_df
    
    def generate_summary_stats(self, df):
        """
        Generate summary statistics for sentiment analysis results
        
        Args:
            df (pandas.DataFrame): Results dataframe
        
        Returns:
            dict: Summary statistics
        """
        if df is None or df.empty:
            return {}
        
        # Basic statistics
        stats = {
            'total_posts': len(df),
            'date_range': {
                'start': df['created_date'].min(),
                'end': df['created_date'].max()
            },
            'textblob': {
                'distribution': df['textblob_sentiment'].value_counts().to_dict(),
                'average_score': df['textblob_score'].mean()
            },
            'vader': {
                'distribution': df['vader_sentiment'].value_counts().to_dict(),
                'average_score': df['vader_score'].mean()
            },
            'most_positive': {
                'title': df.loc[df['vader_score'].idxmax()]['title'],
                'score': df['vader_score'].max(),
                'reddit_score': df.loc[df['vader_score'].idxmax()]['score']
            },
            'most_negative': {
                'title': df.loc[df['vader_score'].idxmin()]['title'],
                'score': df['vader_score'].min(),
                'reddit_score': df.loc[df['vader_score'].idxmin()]['score']
            }
        }
        
        return stats

    def get_popular_subreddits(self):
        """Return a list of popular subreddits"""
        return [
            "funny", "AskReddit", "gaming", "aww", "Music", "pics", "science", "worldnews",
            "videos", "todayilearned", "news", "movies", "Showerthoughts", "EarthPorn", "food",
            "mildlyinteresting", "space", "sports", "television", "Art", "books", "personalfinance",
            "technology", "programming", "Python", "MachineLearning", "datascience", "artificial",
            "cryptocurrency", "stocks", "investing", "politics", "unpopularopinion", "LifeProTips",
            "explainlikeimfive", "NoStupidQuestions", "AskHistorians", "AskScience", "relationship_advice",
            "AmItheAsshole", "tifu", "confession", "self", "depression", "anxiety", "getmotivated",
            "fitness", "loseit", "running", "bodyweightfitness", "nutrition", "recipes", "MealPrepSunday",
            "DIY", "HomeImprovement", "gardening", "camping", "hiking", "travel", "solotravel",
            "backpacking", "photography", "itookapicture", "analog", "streetphotography", "cats",
            "dogs", "AnimalsBeingJerks", "AnimalsBeingBros", "rarepuppers", "Eyebleach", "NatureIsFuckingLit",
            "interestingasfuck", "Damnthatsinteresting", "oddlysatisfying", "mildlyinfuriating",
            "CrappyDesign", "DesignPorn", "battlestations", "pcmasterrace", "buildapc", "mechmarket",
            "Android", "iphone", "apple", "gadgets", "futurology", "singularity", "cyberpunk",
            "startrek", "starwars", "marvelstudios", "DC_Cinematic", "moviedetails", "moviesuggestions",
            "netflix", "television", "gameofthrones", "anime", "manga", "cosplay", "DnD",
            "boardgames", "chess", "soccer", "nfl", "nba", "baseball", "hockey", "mma",
            "formula1", "tennis", "golf", "olympics", "cscareerquestions", "csMajors", "EngineeringStudents"
        ]