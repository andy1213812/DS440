from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class SentimentAnalyzer:
    def __init__(self):
        #Initialize the sentiment analyzer
        self.analyzer = SentimentIntensityAnalyzer()

    def analyze_sentiment(self, text):
        #Analyze sentiment of a given text.
        sentiment_score = self.analyzer.polarity_scores(text)["compound"]
        return sentiment_score
