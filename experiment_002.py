import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
import praw
import pandas as pd
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
from datetime import datetime
import re
import threading
import os
from dotenv import load_dotenv

# Set matplotlib to use TkAgg backend for GUI integration
plt.switch_backend('TkAgg')
load_dotenv()
class RedditSentimentGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Reddit Sentiment Analysis Tool")
        self.root.geometry("1200x800")
        
        # Reddit API Configuration - UPDATE THESE WITH YOUR CREDENTIALS
        self.CLIENT_ID = os.getenv('CLIENT_ID')
        self.CLIENT_SECRET = os.getenv('CLIENT_SECRET')
        self.USER_AGENT = os.getenv('USER_AGENT')
        # Initialize variables
        self.reddit = None
        self.vader_analyzer = SentimentIntensityAnalyzer()
        self.top_subreddits = []
        self.analysis_results = None
        
        # Initialize Reddit connection
        self.initialize_reddit()
        
        # Create the GUI
        self.create_widgets()
        
        # Load top subreddits in background
        self.load_top_subreddits()

    def initialize_reddit(self):
        """Initialize Reddit API connection"""
        try:
            self.reddit = praw.Reddit(
                client_id=self.CLIENT_ID,
                client_secret=self.CLIENT_SECRET,
                user_agent=self.USER_AGENT
            )
            # Test the connection
            self.reddit.user.me()
            print("Reddit API connection successful!")
        except Exception as e:
            print(f"Failed to connect to Reddit API: {str(e)}")
            # You could show a messagebox here if needed
    
    def create_widgets(self):
        """Create all GUI widgets"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="Reddit Sentiment Analysis Tool", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Analysis Configuration Section
        config_frame = ttk.LabelFrame(main_frame, text="Analysis Configuration", padding="10")
        config_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        config_frame.columnconfigure(1, weight=1)
        
        # Subreddit selection
        ttk.Label(config_frame, text="Subreddit:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        
        subreddit_frame = ttk.Frame(config_frame)
        subreddit_frame.grid(row=0, column=1, sticky=(tk.W, tk.E))
        subreddit_frame.columnconfigure(0, weight=1)
        
        self.subreddit_var = tk.StringVar()
        self.subreddit_combo = ttk.Combobox(subreddit_frame, textvariable=self.subreddit_var, width=30)
        self.subreddit_combo.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 10))
        
        ttk.Button(subreddit_frame, text="Refresh Top 100", command=self.load_top_subreddits).grid(row=0, column=1)
        
        # Number of posts
        ttk.Label(config_frame, text="Number of Posts:").grid(row=1, column=0, sticky=tk.W, padx=(0, 10), pady=(10, 0))
        self.num_posts_var = tk.StringVar(value="100")
        num_posts_combo = ttk.Combobox(config_frame, textvariable=self.num_posts_var, width=15)
        num_posts_combo['values'] = ("25", "50", "100", "200", "500", "1000")
        num_posts_combo.grid(row=1, column=1, sticky=tk.W, pady=(10, 0))
        
        # Time filter
        ttk.Label(config_frame, text="Time Filter:").grid(row=2, column=0, sticky=tk.W, padx=(0, 10), pady=(10, 0))
        self.time_filter_var = tk.StringVar(value="week")
        time_filter_combo = ttk.Combobox(config_frame, textvariable=self.time_filter_var, width=15)
        time_filter_combo['values'] = ("hour", "day", "week", "month", "year", "all")
        time_filter_combo.grid(row=2, column=1, sticky=tk.W, pady=(10, 0))
        
        # Analysis button
        self.analyze_button = ttk.Button(config_frame, text="Start Analysis", command=self.start_analysis)
        self.analyze_button.grid(row=3, column=0, columnspan=2, pady=(20, 0))
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(config_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # Status label
        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(config_frame, textvariable=self.status_var)
        self.status_label.grid(row=5, column=0, columnspan=2, pady=(5, 0))
        
        # Results Section
        results_frame = ttk.LabelFrame(main_frame, text="Results", padding="10")
        results_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # Results notebook (tabs)
        self.results_notebook = ttk.Notebook(results_frame)
        self.results_notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Summary tab
        self.summary_frame = ttk.Frame(self.results_notebook)
        self.results_notebook.add(self.summary_frame, text="Summary")
        
        self.summary_text = scrolledtext.ScrolledText(self.summary_frame, height=15, width=80)
        self.summary_text.pack(fill=tk.BOTH, expand=True)
        
        # Visualization tab
        self.viz_frame = ttk.Frame(self.results_notebook)
        self.results_notebook.add(self.viz_frame, text="Visualizations")
        
        # Data tab
        self.data_frame = ttk.Frame(self.results_notebook)
        self.results_notebook.add(self.data_frame, text="Raw Data")
        
        # Export buttons
        export_frame = ttk.Frame(results_frame)
        export_frame.grid(row=1, column=0, pady=(10, 0))
        
        ttk.Button(export_frame, text="Export CSV", command=self.export_csv).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(export_frame, text="Export Summary", command=self.export_summary).pack(side=tk.LEFT)
    
    def load_top_subreddits(self):
        """Load top 100 subreddits"""
        self.status_var.set("Loading top subreddits...")
        
        # Pre-defined top 100 popular subreddits (you can update this list)
        popular_subreddits = [
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
        
        self.top_subreddits = popular_subreddits
        self.subreddit_combo['values'] = popular_subreddits
        self.status_var.set("Top subreddits loaded")

    def clean_text(self, text):
        """Clean and preprocess text for sentiment analysis"""
        if not text:
            return ""
        
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'@\w+|u/\w+|r/\w+', '', text)
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
    
    def get_posts(self, subreddit_name, limit, time_filter):
        """Fetch posts from subreddit"""
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
                
                # Update progress
                progress = (i + 1) / limit * 50  # First 50% for fetching
                self.progress_var.set(progress)
                self.root.update_idletasks()
                
        except Exception as e:
            raise Exception(f"Error fetching posts: {str(e)}")
            
        return posts
    
    def analyze_posts(self, posts):
        """Analyze sentiment of posts"""
        results = []
        
        for i, post in enumerate(posts):
            combined_text = f"{post['title']} {post['text']}"
            clean_combined = self.clean_text(combined_text)
            
            if clean_combined:
                tb_sentiment, tb_score = self.analyze_sentiment_textblob(clean_combined)
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
            
            # Update progress (second 50%)
            progress = 50 + (i + 1) / len(posts) * 50
            self.progress_var.set(progress)
            self.root.update_idletasks()
        
        return pd.DataFrame(results)

    def start_analysis(self):
        """Start the sentiment analysis in a separate thread"""
        if not self.reddit:
            messagebox.showerror("Error", "Reddit API connection not established. Please check your credentials in the code.")
            return
        
        if not self.subreddit_var.get():
            messagebox.showerror("Error", "Please select or enter a subreddit")
            return
        
        # Disable analyze button during analysis
        self.analyze_button.config(state='disabled')
        
        # Start analysis in separate thread to prevent GUI freezing
        thread = threading.Thread(target=self.run_analysis)
        thread.daemon = True
        thread.start()
    
    def run_analysis(self):
        """Run the actual analysis"""
        try:
            subreddit_name = self.subreddit_var.get()
            num_posts = int(self.num_posts_var.get())
            time_filter = self.time_filter_var.get()
            
            self.status_var.set("Fetching posts...")
            self.progress_var.set(0)
            
            # Fetch posts
            posts = self.get_posts(subreddit_name, num_posts, time_filter)
            
            if not posts:
                self.status_var.set("No posts found")
                return
            
            self.status_var.set("Analyzing sentiment...")
            
            # Analyze sentiment
            self.analysis_results = self.analyze_posts(posts)
            
            # Update GUI with results
            self.root.after(0, self.display_results, subreddit_name)
            
        except Exception as e:
            error_msg = str(e)
            self.root.after(0, lambda: messagebox.showerror("Analysis Error", error_msg))
        finally:
            self.root.after(0, lambda: self.analyze_button.config(state='normal'))
            self.root.after(0, lambda: self.progress_var.set(0))
    
    def display_results(self, subreddit_name):
        """Display analysis results in GUI"""
        if self.analysis_results is None or self.analysis_results.empty:
            return
        
        df = self.analysis_results
        
        # Generate summary
        summary = self.generate_summary_text(df, subreddit_name)
        self.summary_text.delete(1.0, tk.END)
        self.summary_text.insert(1.0, summary)
        
        # Create visualizations
        self.create_visualizations(df, subreddit_name)
        
        # Display raw data
        self.display_raw_data(df)
        
        self.status_var.set(f"Analysis complete for r/{subreddit_name}")
    
    def generate_summary_text(self, df, subreddit_name):
        """Generate summary text"""
        summary = f"SENTIMENT ANALYSIS REPORT FOR r/{subreddit_name}\n"
        summary += "=" * 60 + "\n\n"
        
        summary += f"Total posts analyzed: {len(df)}\n"
        summary += f"Date range: {df['created_date'].min()} to {df['created_date'].max()}\n\n"
        
        # TextBlob sentiment distribution
        summary += "TextBlob Sentiment Distribution:\n"
        tb_counts = df['textblob_sentiment'].value_counts()
        for sentiment, count in tb_counts.items():
            percentage = (count / len(df)) * 100
            summary += f"  {sentiment.capitalize()}: {count} posts ({percentage:.1f}%)\n"
        
        summary += "\nVADER Sentiment Distribution:\n"
        vader_counts = df['vader_sentiment'].value_counts()
        for sentiment, count in vader_counts.items():
            percentage = (count / len(df)) * 100
            summary += f"  {sentiment.capitalize()}: {count} posts ({percentage:.1f}%)\n"
        
        summary += f"\nAverage Sentiment Scores:\n"
        summary += f"  TextBlob average: {df['textblob_score'].mean():.3f}\n"
        summary += f"  VADER average: {df['vader_score'].mean():.3f}\n"
        
        # Most positive and negative posts
        summary += f"\nMost Positive Post (VADER):\n"
        most_positive = df.loc[df['vader_score'].idxmax()]
        summary += f"  Title: {most_positive['title'][:100]}...\n"
        summary += f"  Score: {most_positive['vader_score']:.3f}\n"
        summary += f"  Reddit Score: {most_positive['score']}\n"
        
        summary += f"\nMost Negative Post (VADER):\n"
        most_negative = df.loc[df['vader_score'].idxmin()]
        summary += f"  Title: {most_negative['title'][:100]}...\n"
        summary += f"  Score: {most_negative['vader_score']:.3f}\n"
        summary += f"  Reddit Score: {most_negative['score']}\n"
        
        return summary
    
    def create_visualizations(self, df, subreddit_name):
        """Create and display visualizations"""
        # Clear previous visualizations
        for widget in self.viz_frame.winfo_children():
            widget.destroy()
        
        # Create matplotlib figure
        try:
            plt.style.use('default')  # Use default style instead of seaborn
        except:
            pass  # If style doesn't exist, continue with default
            
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f'Sentiment Analysis Results for r/{subreddit_name}', fontsize=14)
        
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
        
        # Embed plot in tkinter
        canvas = FigureCanvasTkAgg(fig, self.viz_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def display_raw_data(self, df):
        """Display raw data in a treeview"""
        # Clear previous data
        for widget in self.data_frame.winfo_children():
            widget.destroy()
        
        # Create treeview
        tree_frame = ttk.Frame(self.data_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True)
        
        columns = ['title', 'textblob_sentiment', 'textblob_score', 'vader_sentiment', 'vader_score', 'score']
        tree = ttk.Treeview(tree_frame, columns=columns, show='headings', height=15)
        
        # Define headings
        for col in columns:
            tree.heading(col, text=col.replace('_', ' ').title())
            tree.column(col, width=150)
        
        # Add data
        for _, row in df.iterrows():
            title = row['title'][:50] + '...' if len(row['title']) > 50 else row['title']
            tree.insert('', tk.END, values=(
                title,
                row['textblob_sentiment'],
                f"{row['textblob_score']:.3f}",
                row['vader_sentiment'],
                f"{row['vader_score']:.3f}",
                row['score']
            ))
        
        # Add scrollbars
        v_scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=tree.yview)
        h_scrollbar = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL, command=tree.xview)
        tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def export_csv(self):
        """Export results to CSV"""
        if self.analysis_results is None:
            messagebox.showwarning("No Data", "No analysis results to export")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if filename:
            self.analysis_results.to_csv(filename, index=False)
            messagebox.showinfo("Success", f"Data exported to {filename}")
    
    def export_summary(self):
        """Export summary to text file"""
        if self.analysis_results is None:
            messagebox.showwarning("No Data", "No analysis results to export")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if filename:
            with open(filename, 'w') as f:
                f.write(self.summary_text.get(1.0, tk.END))
            messagebox.showinfo("Success", f"Summary exported to {filename}")

def main():
    root = tk.Tk()
    app = RedditSentimentGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()