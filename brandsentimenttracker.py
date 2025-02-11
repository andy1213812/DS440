import tweepy
import pandas as pd
from sentimentanalysis import SentimentAnalyzer

# Replace these with your credentials (These are Emi's credentials for x api)
BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAAAo2ygEAAAAA5QimShfUWGzon0ZbN4DwcnDFF7E%3DR4y52TOhLTUO882RDNCrhWJreS2jg03r2WjccdQ69ngmDWahGV"

# Authenticate
client = tweepy.Client(bearer_token=BEARER_TOKEN)

# Define query (replace 'Nike' with your brand)
query = "Nike -is:retweet lang:en"

# Fetch tweets
tweets = client.search_recent_tweets(query=query, tweet_fields=["created_at", "text"], max_results=10)

# Store tweets in a list
tweet_data = []
if tweets.data:
    sentiment_analyzer = SentimentAnalyzer()  # Initialize sentiment analyzer
    for tweet in tweets.data:
        sentiment_score = sentiment_analyzer.analyze_sentiment(tweet.text)
        tweet_data.append({
            "timestamp": tweet.created_at,
            "text": tweet.text,
            "sentiment_score": sentiment_score
        })

# Convert to DataFrame for further analysis
df = pd.DataFrame(tweet_data)
print(df.head())