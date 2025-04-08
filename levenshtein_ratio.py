import pandas as pd
import Levenshtein

# Load your Excel file
df = pd.read_excel("440dataset.xlsx")

# Get original texts
originals = df[df['manipulation_level'] == 0].set_index(['sample_id', 'length_category'])['text_content']

# Compute true Levenshtein metrics
def calc_levenshtein(row):
    key = (row['sample_id'], row['length_category'])
    try:
        original = originals.loc[key]
        modified = row['text_content']
        distance = Levenshtein.distance(original, modified)
        similarity = 1 - (distance / max(len(original), len(modified)))
        return pd.Series([distance, round(similarity * 100, 2)])
    except KeyError:
        return pd.Series([None, None])

# Apply and store results
df[['levenshtein_distance', 'similarity_percent']] = df.apply(calc_levenshtein, axis=1)

# Save to file
df.to_excel("440dataset_with_true_levenshtein.xlsx", index=False)

print("✅ Done! Levenshtein scores saved to 440dataset_with_true_levenshtein.xlsx")
