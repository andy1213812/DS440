import pandas as pd
from keybert import KeyBERT
from collections import Counter
import matplotlib.pyplot as plt
import re

def load_data(filepath):
    df = pd.read_excel(filepath)
    df.dropna(subset=['text_content', 'tool_classification'], inplace=True)
    return df

def extract_keywords_ai_only(df, top_n=20):
    # Filter for rows where the tool classified the text as AI
    ai_texts = df[df['tool_classification'] == 'AI']['text_content'].tolist()
    
    # Join all AI texts into a single document
    corpus = " ".join(ai_texts)
    
    # Initialize KeyBERT
    kw_model = KeyBERT('all-MiniLM-L6-v2')

    # Extract top keywords (1- to 2-gram phrases)
    keywords = kw_model.extract_keywords(
        corpus,
        keyphrase_ngram_range=(1, 2),
        stop_words='english',
        top_n=top_n
    )

    print(f"\n🧠 Top {top_n} KeyBERT Keywords Used by AI:")
    for kw, score in keywords:
        print(f"{kw}: {score:.4f}")
    
    return keywords

def show_frequent_words(df, top_n=20):
    # Tokenize and clean AI-classified texts
    ai_texts = df[df['tool_classification'] == 'AI']['text_content'].tolist()
    words = " ".join(ai_texts).lower()
    words = re.findall(r'\b[a-z]{3,}\b', words)  # words with 3+ letters

    counter = Counter(words)
    common_words = counter.most_common(top_n)

    print(f"\n📊 Top {top_n} Frequent Words in AI-Classified Texts:")
    for word, freq in common_words:
        print(f"{word}: {freq}")
    
    return common_words

def visualize_keywords(keywords):
    words = [w for w, _ in keywords]
    scores = [s for _, s in keywords]

    plt.figure(figsize=(10, 5))
    plt.barh(words[::-1], scores[::-1], color='skyblue')
    plt.xlabel('Relevance Score')
    plt.title('Top AI-Preferred Keywords (KeyBERT)')
    plt.tight_layout()
    plt.savefig("ai_keywords.png")
    plt.show()

def main():
    filepath = '440dataset.xlsx'  # Adjust if needed
    df = load_data(filepath)

    keybert_keywords = extract_keywords_ai_only(df, top_n=20)
    freq_keywords = show_frequent_words(df, top_n=20)
    
    visualize_keywords(keybert_keywords)

if __name__ == '__main__':
    main()
