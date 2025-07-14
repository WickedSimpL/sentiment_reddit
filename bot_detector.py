import praw
import re
from datetime import datetime, timezone
from collections import Counter
import statistics
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Download required NLTK data (run once)
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

class RedditBotDetector:
    def __init__(self, client_id, client_secret, user_agent):
        """
        Initialize the Reddit Bot Detector
        
        Args:
            client_id: Reddit API client ID
            client_secret: Reddit API client secret
            user_agent: User agent string for API requests
        """
        self.reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent
        )
        self.stop_words = set(stopwords.words('english'))
        
    def analyze_user(self, username, limit=100):
        """
        Analyze a Reddit user for bot-like behavior
        
        Args:
            username: Reddit username to analyze
            limit: Number of recent posts/comments to analyze
            
        Returns:
            Dictionary with bot likelihood score and analysis details
        """
        try:
            user = self.reddit.redditor(username)
            user.comment_karma  # Test if user exists
        except:
            return {"error": "User not found or suspended"}
        
        # Collect user's recent activity
        comments = list(user.comments.new(limit=limit))
        submissions = list(user.submissions.new(limit=limit))
        
        # Perform various analyses
        timing_score = self._analyze_posting_patterns(comments + submissions)
        content_score = self._analyze_content_patterns(comments, submissions)
        activity_score = self._analyze_activity_patterns(comments, submissions)
        account_score = self._analyze_account_metrics(user)
        
        # Calculate overall bot likelihood (0-100)
        bot_likelihood = (
            timing_score * 0.25 +
            content_score * 0.35 +
            activity_score * 0.25 +
            account_score * 0.15
        )
        
        return {
            "username": username,
            "bot_likelihood": round(bot_likelihood, 2),
            "timing_score": round(timing_score, 2),
            "content_score": round(content_score, 2),
            "activity_score": round(activity_score, 2),
            "account_score": round(account_score, 2),
            "risk_level": self._get_risk_level(bot_likelihood),
            "details": {
                "posts_analyzed": len(submissions),
                "comments_analyzed": len(comments),
                "account_age_days": (datetime.now(timezone.utc) - datetime.fromtimestamp(user.created_utc, tz=timezone.utc)).days,
                "karma": user.comment_karma + user.link_karma
            }
        }
    
    def analyze_post(self, post_url):
        """
        Analyze a specific Reddit post and its comments for bot activity
        
        Args:
            post_url: URL of the Reddit post
            
        Returns:
            Dictionary with analysis of post and suspicious comments
        """
        submission = self.reddit.submission(url=post_url)
        submission.comments.replace_more(limit=0)
        
        # Analyze OP
        op_analysis = self.analyze_user(submission.author.name if submission.author else "[deleted]", limit=50)
        
        # Analyze comments
        suspicious_comments = []
        for comment in submission.comments.list()[:100]:  # Analyze top 100 comments
            if comment.author:
                commenter_analysis = self._quick_user_check(comment.author.name)
                if commenter_analysis['bot_likelihood'] > 60:
                    suspicious_comments.append({
                        "author": comment.author.name,
                        "bot_likelihood": commenter_analysis['bot_likelihood'],
                        "comment_preview": comment.body[:100] + "..." if len(comment.body) > 100 else comment.body
                    })
        
        return {
            "post_title": submission.title,
            "post_author_analysis": op_analysis,
            "suspicious_comments": suspicious_comments,
            "total_comments_analyzed": min(100, len(submission.comments.list())),
            "bot_comment_ratio": len(suspicious_comments) / min(100, len(submission.comments.list())) if submission.comments.list() else 0
        }
    
    def _analyze_posting_patterns(self, posts):
        """Analyze timing patterns of posts/comments"""
        if not posts:
            return 50  # Neutral score if no data
        
        # Extract posting times
        timestamps = [post.created_utc for post in posts]
        if len(timestamps) < 2:
            return 50
        
        # Calculate time intervals between posts
        intervals = []
        for i in range(1, len(timestamps)):
            intervals.append(timestamps[i-1] - timestamps[i])
        
        # Check for suspicious patterns
        score = 0
        
        # 1. Regular intervals (bot-like)
        if len(intervals) > 5:
            std_dev = statistics.stdev(intervals)
            mean_interval = statistics.mean(intervals)
            cv = std_dev / mean_interval if mean_interval > 0 else 0
            if cv < 0.3:  # Low coefficient of variation suggests regular posting
                score += 30
        
        # 2. Rapid-fire posting
        rapid_posts = sum(1 for i in intervals if i < 60)  # Posts within 1 minute
        if rapid_posts / len(intervals) > 0.3:
            score += 40
        
        # 3. 24/7 activity pattern
        hours = [datetime.fromtimestamp(ts, tz=timezone.utc).hour for ts in timestamps]
        hour_distribution = Counter(hours)
        if len(hour_distribution) > 20:  # Active in almost all hours
            score += 30
        
        return min(score, 100)
    
    def _analyze_content_patterns(self, comments, submissions):
        """Analyze content for bot-like patterns"""
        if not comments and not submissions:
            return 50
        
        score = 0
        
        # Analyze comments
        if comments:
            comment_texts = [c.body for c in comments]
            
            # 1. Repetitive content
            unique_comments = len(set(comment_texts))
            print(f"Num Unqiue Comments {unique_comments}")
            if unique_comments / len(comment_texts) < 0.5:
                score += 30
            
            # 2. Generic responses
            generic_phrases = [
                "this", "nice", "thanks for sharing", "great post",
                "i agree", "well said", "exactly", "this is the way"
            ]
            generic_count = sum(1 for text in comment_texts if any(phrase in text.lower() for phrase in generic_phrases))
            if generic_count / len(comment_texts) > 0.4:
                score += 25
            
            # 3. Low vocabulary diversity
            all_words = []
            for text in comment_texts:
                words = word_tokenize(text.lower())
                words = [w for w in words if w not in self.stop_words and w not in string.punctuation]
                all_words.extend(words)
            
            if all_words:
                vocab_diversity = len(set(all_words)) / len(all_words)
                if vocab_diversity < 0.1:
                    score += 25
        
        # Analyze submissions
        if submissions:
            # Check for repost patterns
            titles = [s.title for s in submissions]
            unique_titles = len(set(titles))
            if unique_titles / len(titles) < 0.7:
                score += 20
        
        return min(score, 100)
    
    def _analyze_activity_patterns(self, comments, submissions):
        """Analyze activity patterns for bot-like behavior"""
        score = 0
        
        # 1. Subreddit concentration
        subreddits = []
        for item in comments + submissions:
            subreddits.append(item.subreddit.display_name)
        
        if subreddits:
            subreddit_counts = Counter(subreddits)
            top_subreddit_ratio = subreddit_counts.most_common(1)[0][1] / len(subreddits)
            if top_subreddit_ratio > 0.7:  # Highly concentrated in one subreddit
                score += 30
        
        # 2. Karma farming patterns
        if submissions:
            repost_subs = ['FreeKarma4U', 'FreeKarma4You', 'karma']
            karma_farming = sum(1 for s in submissions if any(sub in s.subreddit.display_name for sub in repost_subs))
            if karma_farming > 0:
                score += 40
        
        # 3. Response time to posts (for comments)
        if comments:
            quick_responses = 0
            for comment in comments:
                try:
                    parent = comment.parent()
                    if hasattr(parent, 'created_utc'):
                        response_time = comment.created_utc - parent.created_utc
                        if response_time < 60:  # Responded within 1 minute
                            quick_responses += 1
                except:
                    pass
            
            if quick_responses / len(comments) > 0.5:
                score += 30
        
        return min(score, 100)
    
    def _analyze_account_metrics(self, user):
        """Analyze account-level metrics"""
        score = 0
        
        # 1. Account age vs karma ratio
        account_age_days = (datetime.now(timezone.utc) - datetime.fromtimestamp(user.created_utc, tz=timezone.utc)).days
        total_karma = user.comment_karma + user.link_karma
        
        if account_age_days < 30 and total_karma > 10000:
            score += 40  # Suspiciously high karma for new account
        
        # 2. Username patterns
        username = user.name
        # Check for random-looking usernames
        if re.match(r'^[A-Za-z]+\d{4,}$', username):  # Like "User1234567"
            score += 30
        if re.match(r'^[A-Za-z]+[-_][A-Za-z]+[-_]\d+$', username):  # Like "Word-Word-123"
            score += 30
        
        # 3. Profile completeness
        if not user.subreddit.get('public_description', ''):
            score += 10  # No profile description
        
        return min(score, 100)
    
    def _quick_user_check(self, username):
        """Quick bot check for bulk analysis"""
        try:
            user = self.reddit.redditor(username)
            recent_comments = list(user.comments.new(limit=10))
            
            # Quick heuristics
            score = 0
            
            # Check posting frequency
            if len(recent_comments) >= 2:
                time_diff = recent_comments[0].created_utc - recent_comments[-1].created_utc
                if time_diff < 600 and len(recent_comments) == 10:  # 10 comments in 10 minutes
                    score += 50
            
            # Check for generic comments
            generic_count = sum(1 for c in recent_comments if len(c.body.split()) < 5)
            if generic_count / len(recent_comments) > 0.7:
                score += 50
            
            return {"bot_likelihood": min(score, 100)}
        except:
            return {"bot_likelihood": 0}
    
    def _get_risk_level(self, score):
        """Convert bot likelihood score to risk level"""
        if score >= 80:
            return "HIGH"
        elif score >= 60:
            return "MEDIUM"
        elif score >= 40:
            return "LOW"
        else:
            return "MINIMAL"


# Example usage
if __name__ == "__main__":
    # Initialize detector with your Reddit API credentials
    detector = RedditBotDetector(
        client_id=os.getenv('CLIENT_ID'),
        client_secret=os.getenv('CLIENT_SECRET'),
        user_agent=os.getenv('USER_AGENT')
    )
    
    # Example 2: Analyze a post and its comments
    print("\n\nAnalyzing post...")
    post_analysis = detector.analyze_post("https://www.reddit.com/r/politics/comments/1l9qbdx/trumps_birthday_parade_may_be_cancelled_over/")
    print(f"\nPost Analysis: {post_analysis['post_title']}")
    print(f"OP Bot Likelihood: {post_analysis['post_author_analysis'].get('bot_likelihood', 'N/A')}%")
    print(f"Suspicious Comments Found: {len(post_analysis['suspicious_comments'])}")
    print(f"Bot Comment Ratio: {post_analysis['bot_comment_ratio']:.2%}")