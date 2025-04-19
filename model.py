import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from Levenshtein import distance as levenshtein_distance
from collections import Counter

# Load data
dataset = pd.read_excel('440dataset.xlsx')

# Load pre-trained SBERT model for semantic similarity
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Load spaCy model for POS tagging
nlp = spacy.load("en_core_web_sm")

# Function to compute Levenshtein distance
def compute_levenshtein(text1, text2):
    return levenshtein_distance(text1, text2)

# Function to compute cosine similarity using SBERT
def compute_cosine_similarity(text1, text2):
    embeddings1 = model.encode([text1])
    embeddings2 = model.encode([text2])
    return cosine_similarity(embeddings1, embeddings2)[0][0]

# Function to perform POS tagging and return POS tags
def get_pos_tags(text):
    doc = nlp(text)
    pos_tags = [token.pos_ for token in doc]
    return pos_tags

# Function to compute word frequency
def compute_word_frequency(text):
    words = text.split()
    return dict(Counter(words))

# Function to extract structural changes by comparing POS tags
def compare_pos_tags(original_text, paraphrased_text):
    original_pos = get_pos_tags(original_text)
    paraphrased_pos = get_pos_tags(paraphrased_text)
    pos_changes = sum(1 for o, p in zip(original_pos, paraphrased_pos) if o != p)
    return pos_changes

# Loop through dataset and perform analysis
results = []

# Iterate over dataset
for _, row in dataset.iterrows():
    original_text = row['text_content']  # Original text
    manipulation_level = row['manipulation_level']
    
    # Only process rows where manipulation level is 3, 6, or 9
    if manipulation_level in [3, 6, 9]:
        # Find the paraphrased version with manipulation level 0
        filtered_data = dataset[(dataset['sample_id'] == row['sample_id']) & (dataset['manipulation_level'] == 0)]
        
        if not filtered_data.empty:
            paraphrased_text = filtered_data['text_content'].values[0]
        else:
            paraphrased_text = None  # No paraphrased version found
            print(f"No paraphrased text found for sample_id {row['sample_id']} at manipulation_level 0")
        
        if paraphrased_text:  # Only perform analysis if paraphrased text exists
            # Compute Levenshtein distance
            lev_dist = compute_levenshtein(original_text, paraphrased_text)
            
            # Compute cosine similarity
            cos_sim = compute_cosine_similarity(original_text, paraphrased_text)
            
            # Compute POS changes
            pos_changes = compare_pos_tags(original_text, paraphrased_text)
            
            # Compute word frequency changes
            original_word_freq = compute_word_frequency(original_text)
            paraphrased_word_freq = compute_word_frequency(paraphrased_text)
            word_freq_changes = sum(abs(original_word_freq.get(word, 0) - paraphrased_word_freq.get(word, 0)) for word in original_word_freq)
            
            # Store results
            results.append({
                'sample_id': row['sample_id'],
                'manipulation_level': manipulation_level,
                'levenshtein_distance': lev_dist,
                'cosine_similarity': cos_sim,
                'pos_changes': pos_changes,
                'word_freq_changes': word_freq_changes
            })
        else:
            # Store results as None for missing paraphrased text
            results.append({
                'sample_id': row['sample_id'],
                'manipulation_level': manipulation_level,
                'levenshtein_distance': None,
                'cosine_similarity': None,
                'pos_changes': None,
                'word_freq_changes': None
            })

# Create a DataFrame to analyze results
results_df = pd.DataFrame(results)

# Save the results to CSV
results_df.to_csv('results.csv', index=False)

# Display results
print(results_df)
