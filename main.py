#!/usr/bin/env python3
"""
Reddit Sentiment Analysis Tool

A GUI application for analyzing sentiment in Reddit posts using TextBlob and VADER.
Supports analysis of any subreddit with configurable parameters.

Author: Jack Palmstrom
Date: 2025
"""

import tkinter as tk
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gui_interface import RedditSentimentGUI

def main():
    """Main function to run the Reddit Sentiment Analysis GUI"""
    try:
        # Create the main window
        root = tk.Tk()
        
        # Initialize the application
        app = RedditSentimentGUI(root)
        
        # Start the GUI event loop
        root.mainloop()
        
    except KeyboardInterrupt:
        print("\nApplication interrupted by user")
        sys.exit(0)
    except ImportError as e:
        print(f"Import error: {e}")
        print("Please make sure all required packages are installed:")
        print("pip install praw pandas textblob vaderSentiment matplotlib seaborn python-dotenv")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()