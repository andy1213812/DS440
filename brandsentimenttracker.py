import tweepy

# Replace these with your credentials
BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAAAo2ygEAAAAA5QimShfUWGzon0ZbN4DwcnDFF7E%3DR4y52TOhLTUO882RDNCrhWJreS2jg03r2WjccdQ69ngmDWahGV"

# Authenticate
client = tweepy.Client(bearer_token=BEARER_TOKEN)

# Define query (replace 'Nike' with your brand)
query = "Nike -is:retweet lang:en"

# Fetch tweets
tweets = client.search_recent_tweets(query=query, tweet_fields=["created_at", "text"], max_results=10)

# Print tweets
if tweets.data:
    for tweet in tweets.data:
        print(f"{tweet.created_at}: {tweet.text}\n")
else:
    print("No tweets found.")