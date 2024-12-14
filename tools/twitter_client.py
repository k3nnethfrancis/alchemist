from datetime import datetime
from typing import List, Dict, ClassVar
from pathlib import Path
import json
from mirascope.core import BaseTool

class TwitterTool:
    def __init__(self):
        self.tweets_file = Path("data/tweets.json")
        self.tweets_file.parent.mkdir(exist_ok=True)
        if not self.tweets_file.exists():
            self.tweets_file.write_text(json.dumps([
                {
                    "username": "@TechEnthusiast",
                    "timestamp": "2024-03-15 10:30:00",
                    "content": "Just discovered a new AI framework that's blowing my mind! 🤖 #AI #Tech",
                    "likes": 42,
                    "retweets": 12
                }
            ]))

    def check_feed(self) -> str:
        """Get the most recent tweets from the feed."""
        return self.tweets_file.read_text()

    def write_tweet(self, content: str) -> str:
        """Write a new tweet."""
        tweets = json.loads(self.tweets_file.read_text())
        new_tweet = {
            "username": "@User",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "content": content,
            "likes": 0,
            "retweets": 0
        }
        tweets.append(new_tweet)
        self.tweets_file.write_text(json.dumps(tweets))
        return json.dumps({"status": "success", "tweet": new_tweet})

class CheckTwitterFeed(BaseTool):
    name: ClassVar[str] = "check_twitter_feed"
    description: ClassVar[str] = "Get the most recent tweets from the feed"
    
    def call(self) -> str:
        return TwitterTool().check_feed()

class WriteTwitterTweet(BaseTool):
    name: ClassVar[str] = "write_tweet"
    description: ClassVar[str] = "Write a new tweet"
    parameters: ClassVar[Dict] = {
        "content": {
            "type": "string",
            "description": "The content of the tweet to post"
        }
    }
    content: str
    
    def call(self) -> str:
        return TwitterTool().write_tweet(self.content) 