import praw
import pandas as pd
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import re
import time

class RedditSentimentAnalyzer:
    def __init__(self, client_id, client_secret, user_agent):
        """
        Initialize the Reddit API client and sentiment analyzer
        
        Args:
            client_id (str): Reddit API client ID
            client_secret (str): Reddit API client secret
            user_agent (str): User agent string for Reddit API
        """
        self.reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent
        )
        self.vader_analyzer = SentimentIntensityAnalyzer()
        
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
    
    def get_posts(self, subreddit_name, limit=100, time_filter='week'):
        """
        Fetch posts from a subreddit
        
        Args:
            subreddit_name (str): Name of the subreddit
            limit (int): Number of posts to fetch
            time_filter (str): Time filter ('hour', 'day', 'week', 'month', 'year', 'all')
        
        Returns:
            list: List of post dictionaries
        """
        posts = []
        
        try:
            subreddit = self.reddit.subreddit(subreddit_name)
            
            # Get top posts from the specified time period
            for post in subreddit.top(time_filter=time_filter, limit=limit):
                posts.append({
                    'title': post.title,
                    'text': post.selftext,
                    'score': post.score,
                    'num_comments': post.num_comments,
                    'created_utc': datetime.fromtimestamp(post.created_utc),
                    'url': post.url,
                    'id': post.id
                })
                
        except Exception as e:
            print(f"Error fetching posts: {str(e)}")
            
        return posts
    
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
    
    def analyze_subreddit(self, subreddit_name, limit=100, time_filter='week'):
        """
        Perform complete sentiment analysis on a subreddit
        
        Args:
            subreddit_name (str): Name of the subreddit
            limit (int): Number of posts to analyze
            time_filter (str): Time filter for posts
        
        Returns:
            pandas.DataFrame: DataFrame with posts and sentiment analysis
        """
        print(f"Fetching posts from r/{subreddit_name}...")
        posts = self.get_posts(subreddit_name, limit, time_filter)
        
        if not posts:
            print("No posts found!")
            return None
        
        print(f"Analyzing sentiment for {len(posts)} posts...")
        
        results = []
        for post in posts:
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
        
        return pd.DataFrame(results)
    
    def generate_report(self, df, subreddit_name):
        """Generate a comprehensive sentiment analysis report"""
        if df is None or df.empty:
            print("No data to analyze!")
            return
        
        print(f"\n=== SENTIMENT ANALYSIS REPORT FOR r/{subreddit_name} ===\n")
        
        # Basic statistics
        print(f"Total posts analyzed: {len(df)}")
        print(f"Date range: {df['created_date'].min()} to {df['created_date'].max()}")
        
        # TextBlob sentiment distribution
        print("\n--- TextBlob Sentiment Distribution ---")
        tb_counts = df['textblob_sentiment'].value_counts()
        for sentiment, count in tb_counts.items():
            percentage = (count / len(df)) * 100
            print(f"{sentiment.capitalize()}: {count} posts ({percentage:.1f}%)")
        
        # VADER sentiment distribution
        print("\n--- VADER Sentiment Distribution ---")
        vader_counts = df['vader_sentiment'].value_counts()
        for sentiment, count in vader_counts.items():
            percentage = (count / len(df)) * 100
            print(f"{sentiment.capitalize()}: {count} posts ({percentage:.1f}%)")
        
        # Average scores
        print(f"\n--- Average Sentiment Scores ---")
        print(f"TextBlob average: {df['textblob_score'].mean():.3f}")
        print(f"VADER average: {df['vader_score'].mean():.3f}")
        
        # Most positive and negative posts
        print(f"\n--- Most Positive Post (VADER) ---")
        most_positive = df.loc[df['vader_score'].idxmax()]
        print(f"Title: {most_positive['title']}")
        print(f"Score: {most_positive['vader_score']:.3f}")
        print(f"Reddit Score: {most_positive['score']}")
        
        print(f"\n--- Most Negative Post (VADER) ---")
        most_negative = df.loc[df['vader_score'].idxmin()]
        print(f"Title: {most_negative['title']}")
        print(f"Score: {most_negative['vader_score']:.3f}")
        print(f"Reddit Score: {most_negative['score']}")
    
    def create_visualizations(self, df, subreddit_name):
        """Create visualizations for sentiment analysis results"""
        if df is None or df.empty:
            return
        
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Sentiment Analysis Results for r/{subreddit_name}', fontsize=16)
        
        # TextBlob sentiment distribution
        tb_counts = df['textblob_sentiment'].value_counts()
        axes[0, 0].pie(tb_counts.values, labels=tb_counts.index, autopct='%1.1f%%')
        axes[0, 0].set_title('TextBlob Sentiment Distribution')
        
        # VADER sentiment distribution
        vader_counts = df['vader_sentiment'].value_counts()
        axes[0, 1].pie(vader_counts.values, labels=vader_counts.index, autopct='%1.1f%%')
        axes[0, 1].set_title('VADER Sentiment Distribution')
        
        # Sentiment score distributions
        axes[1, 0].hist(df['textblob_score'], bins=20, alpha=0.7, color='blue')
        axes[1, 0].set_title('TextBlob Score Distribution')
        axes[1, 0].set_xlabel('Sentiment Score')
        axes[1, 0].set_ylabel('Frequency')
        
        axes[1, 1].hist(df['vader_score'], bins=20, alpha=0.7, color='red')
        axes[1, 1].set_title('VADER Score Distribution')
        axes[1, 1].set_xlabel('Sentiment Score')
        axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(f'sentiment_analysis_{subreddit_name}.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Main function to run the sentiment analysis"""
    
    # Reddit API credentials - You need to get these from https://www.reddit.com/prefs/apps
    CLIENT_ID = "NPb6ZunKMWp-PmPZz5IyKg"
    CLIENT_SECRET = "0jtlsZBhUsGn9nmoi6znnN8NYxlO_A"
    USER_AGENT = "SentimentAnalyzer/1.0 by simpleJack404"
    
    # Initialize the analyzer
    analyzer = RedditSentimentAnalyzer(CLIENT_ID, CLIENT_SECRET, USER_AGENT)
    
    # Configuration
    subreddit_name = input("Enter subreddit name (without r/): ").strip()
    
    try:
        num_posts = int(input("Enter number of posts to analyze (default 100): ") or "100")
    except ValueError:
        num_posts = 100
    
    time_filter = input("Enter time filter (hour/day/week/month/year/all, default 'week'): ").strip() or "week"
    
    # Run analysis
    try:
        df = analyzer.analyze_subreddit(subreddit_name, num_posts, time_filter)
        
        if df is not None:
            # Generate report
            analyzer.generate_report(df, subreddit_name)
            
            # Create visualizations
            analyzer.create_visualizations(df, subreddit_name)
            
            # Save results to CSV
            filename = f"sentiment_analysis_{subreddit_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(filename, index=False)
            print(f"\nResults saved to {filename}")
            
        else:
            print("Failed to analyze subreddit. Please check the subreddit name and try again.")
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()