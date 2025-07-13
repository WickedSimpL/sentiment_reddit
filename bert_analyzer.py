import praw
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class RedditSentimentAnalyzer:
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
            
            # Test the connection
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
        
        # Initialize BERT model for sentiment analysis
        print("\nLoading BERT model...")
        self.model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        
        # Set device (GPU if available)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.model.to(self.device)
        self.model.eval()
        print("✓ BERT model loaded successfully")
        
    def fetch_subreddit_posts(self, subreddit_name, limit=100, time_filter='week'):
        """
        Fetch posts from a specific subreddit with error handling
        """
        try:
            subreddit = self.reddit.subreddit(subreddit_name)
            
            # Verify subreddit exists
            try:
                _ = subreddit.display_name
            except:
                raise ValueError(f"Subreddit r/{subreddit_name} not found or is private")
            
            posts_data = []
            
            # Fetch top posts
            for post in subreddit.top(time_filter=time_filter, limit=limit):
                posts_data.append({
                    'title': post.title,
                    'text': post.selftext,
                    'score': post.score,
                    'num_comments': post.num_comments,
                    'created_utc': datetime.fromtimestamp(post.created_utc),
                    'id': post.id,
                    'url': post.url
                })
            
            return pd.DataFrame(posts_data)
            
        except Exception as e:
            print(f"Error fetching posts: {str(e)}")
            raise
    
    def fetch_comments(self, post_id, limit=500):
        """
        Fetch comments from a specific post
        """
        submission = self.reddit.submission(id=post_id)
        submission.comments.replace_more(limit=0)
        
        comments = []
        for comment in submission.comments.list()[:limit]:
            if comment.body and comment.body != '[deleted]':
                comments.append(comment.body)
        
        return comments
    
    def predict_sentiment(self, texts, batch_size=64):
        """
        Predict sentiment for a list of texts using BERT
        """
        if not texts:
            return [], []
            
        sentiments = []
        confidences = []
        
        # Process in batches
        for i in tqdm(range(0, len(texts), batch_size), desc="Analyzing sentiment"):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize texts
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )
            
            # Move to device
            input_ids = encoded['input_ids'].to(self.device)
            attention_mask = encoded['attention_mask'].to(self.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                
                # Get sentiment scores (1-5) and confidence
                sentiment_scores = torch.argmax(predictions, dim=-1) + 1
                confidence_scores = torch.max(predictions, dim=-1).values
                
                sentiments.extend(sentiment_scores.cpu().numpy().tolist())
                confidences.extend(confidence_scores.cpu().numpy().tolist())
        
        return sentiments, confidences
    
    def analyze_subreddit(self, subreddit_name, num_posts=500, include_comments=True):
        """
        Perform complete sentiment analysis on a subreddit
        """
        print(f"\nFetching posts from r/{subreddit_name}...")
        posts_df = self.fetch_subreddit_posts(subreddit_name, limit=num_posts)
        
        if posts_df.empty:
            print(f"No posts found in r/{subreddit_name}")
            return None
        
        print(f"Found {len(posts_df)} posts")
        
        # Combine title and text for analysis
        posts_df['full_text'] = posts_df['title'] + ' ' + posts_df['text'].fillna('')
        
        # Analyze post sentiments
        print("Analyzing post sentiments...")
        post_sentiments, post_confidences = self.predict_sentiment(posts_df['full_text'].tolist())
        posts_df['sentiment'] = post_sentiments
        posts_df['confidence'] = post_confidences
        
        results = {
            'posts_df': posts_df,
            'post_sentiments': post_sentiments,
            'average_post_sentiment': np.mean(post_sentiments),
            'sentiment_distribution': pd.Series(post_sentiments).value_counts().to_dict()
        }
        
        # Analyze comments if requested
        if include_comments:
            print("Fetching and analyzing comments...")
            all_comments = []
            comment_sentiments = []
            
            for post_id in tqdm(posts_df['id'].head(10), desc="Fetching comments"):
                try:
                    comments = self.fetch_comments(post_id, limit=20)
                    all_comments.extend(comments)
                except:
                    continue
            
            if all_comments:
                comment_sentiments, comment_confidences = self.predict_sentiment(all_comments)
                results['comment_sentiments'] = comment_sentiments
                results['average_comment_sentiment'] = np.mean(comment_sentiments)
                results['comment_distribution'] = pd.Series(comment_sentiments).value_counts().to_dict()
        
        return results
    
    def visualize_results(self, results, subreddit_name):
        """
        Create visualizations for sentiment analysis results
        """
        if not results:
            print("No results to visualize")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Sentiment Analysis for r/{subreddit_name}', fontsize=16)
        
        # Post sentiment distribution
        sentiment_counts = pd.Series(results['post_sentiments']).value_counts().sort_index()
        axes[0, 0].bar(sentiment_counts.index, sentiment_counts.values)
        axes[0, 0].set_xlabel('Sentiment Score')
        axes[0, 0].set_ylabel('Number of Posts')
        axes[0, 0].set_title('Post Sentiment Distribution')
        axes[0, 0].set_xticks([1, 2, 3, 4, 5])
        axes[0, 0].set_xticklabels(['Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive'])
        plt.setp(axes[0, 0].xaxis.get_majorticklabels(), rotation=45)
        
        # Sentiment vs engagement
        posts_df = results['posts_df']
        axes[0, 1].scatter(posts_df['sentiment'], posts_df['score'], alpha=0.6)
        axes[0, 1].set_xlabel('Sentiment Score')
        axes[0, 1].set_ylabel('Post Score')
        axes[0, 1].set_title('Sentiment vs Post Engagement')
        
        # Time series of sentiment
        posts_df['date'] = pd.to_datetime(posts_df['created_utc']).dt.date
        daily_sentiment = posts_df.groupby('date')['sentiment'].mean()
        if not daily_sentiment.empty:
            axes[1, 0].plot(daily_sentiment.index, daily_sentiment.values, marker='o')
            axes[1, 0].set_xlabel('Date')
            axes[1, 0].set_ylabel('Average Sentiment')
            axes[1, 0].set_title('Sentiment Over Time')
            plt.setp(axes[1, 0].xaxis.get_majorticklabels(), rotation=45)
        
        # Comment sentiment distribution (if available)
        if 'comment_sentiments' in results and results['comment_sentiments']:
            comment_counts = pd.Series(results['comment_sentiments']).value_counts().sort_index()
            axes[1, 1].bar(comment_counts.index, comment_counts.values, color='orange')
            axes[1, 1].set_xlabel('Sentiment Score')
            axes[1, 1].set_ylabel('Number of Comments')
            axes[1, 1].set_title('Comment Sentiment Distribution')
            axes[1, 1].set_xticks([1, 2, 3, 4, 5])
            axes[1, 1].set_xticklabels(['Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive'])
            plt.setp(axes[1, 1].xaxis.get_majorticklabels(), rotation=45)
        else:
            axes[1, 1].text(0.5, 0.5, 'No comment data available', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
        
        plt.tight_layout()
        plt.show()
        
        # Print summary statistics
        print(f"\n=== Sentiment Analysis Summary for r/{subreddit_name} ===")
        print(f"Average Post Sentiment: {results['average_post_sentiment']:.2f}")
        print(f"Post Sentiment Distribution: {results['sentiment_distribution']}")
        
        if 'average_comment_sentiment' in results:
            print(f"Average Comment Sentiment: {results['average_comment_sentiment']:.2f}")
            print(f"Comment Sentiment Distribution: {results['comment_distribution']}")


# Create a test script to verify credentials
def test_credentials():
    """Test if Reddit credentials are properly loaded"""
    print("Testing environment variables...")
    
    # Check if .env file exists
    if os.path.exists('.env'):
        print("✓ .env file found")
    else:
        print("✗ .env file not found")
        print("\nCreate a .env file with the following format:")
        print("CLIENT_ID=your_reddit_client_id")
        print("CLIENT_SECRET=your_reddit_client_secret")
        print("USER_AGENT=your_app_name/1.0 by /u/your_reddit_username")
        return False
    
    # Load environment variables
    load_dotenv()
    
    # Check each credential
    client_id = os.getenv('CLIENT_ID')
    client_secret = os.getenv('CLIENT_SECRET')
    user_agent = os.getenv('USER_AGENT')
    
    print(f"\nCLIENT_ID: {'✓ Loaded' if client_id else '✗ Not found'}")
    print(f"CLIENT_SECRET: {'✓ Loaded' if client_secret else '✗ Not found'}")
    print(f"USER_AGENT: {'✓ Loaded' if user_agent else '✗ Not found'}")
    
    if not all([client_id, client_secret, user_agent]):
        print("\n✗ Missing credentials. Please check your .env file.")
        return False
    
    print("\n✓ All credentials loaded successfully")
    return True


# Example usage
if __name__ == "__main__":
    # First test credentials
    if not test_credentials():
        print("\nPlease fix credential issues before running the analyzer.")
        exit(1)
    
    # Load credentials
    CLIENT_ID = os.getenv('CLIENT_ID')
    CLIENT_SECRET = os.getenv('CLIENT_SECRET')
    USER_AGENT = os.getenv('USER_AGENT')
    
    try:
        # Initialize analyzer
        print("\nInitializing Reddit Sentiment Analyzer...")
        analyzer = RedditSentimentAnalyzer(CLIENT_ID, CLIENT_SECRET, USER_AGENT)
        
        # Analyze a subreddit
        subreddit_name = "python"  # Change to your desired subreddit
        results = analyzer.analyze_subreddit(subreddit_name, num_posts=500, include_comments=True)
        
        if results:
            # Visualize results
            analyzer.visualize_results(results, subreddit_name)
            
            # Save results to CSV
            results['posts_df'].to_csv(f'{subreddit_name}_sentiment_analysis.csv', index=False)
            print(f"\nResults saved to {subreddit_name}_sentiment_analysis.csv")
            
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("\nCommon issues:")
        print("1. Invalid CLIENT_ID or CLIENT_SECRET")
        print("2. USER_AGENT format should be: 'bot_name/version by /u/username'")
        print("3. Reddit API may be down or rate limited")
        print("4. Check if your Reddit app type is 'script' (not 'web app')")