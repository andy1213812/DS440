import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sentence_transformers import SentenceTransformer
from sklearn.model_selection import StratifiedKFold

# Step 1: Load and clean the dataset
def load_and_clean_data(filepath):
    df = pd.read_excel(filepath)
    df.drop(index=[54, 55, 56, 57], inplace=True)
    df = df[df['text_type'].isin(['AI', 'human'])].reset_index(drop=True)
    return df

# Step 2: Generate embeddings
def compute_embeddings(df, model_name='all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    df['embedding'] = df['text_content'].apply(lambda x: model.encode(x))
    return df

# Step 3: Train a classifier to predict 'text_type'
def train_classifier(df):
    # Keep only necessary columns
    df = df[['sample_id', 'text_content', 'tool_classification', 'embedding']]

    # Optionally sample 1 row per sample_id to avoid redundancy
    df = df.groupby('sample_id').apply(lambda group: group.sample(1, random_state=42)).reset_index(drop=True)

    X = np.vstack(df['embedding'].values)
    y = df['tool_classification'].values

    # Add tiny noise
    noise = np.random.normal(0, 0.01, X.shape)
    X_noisy = X + noise

    # 5-fold stratified CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    acc_scores = []

    for train_idx, test_idx in skf.split(X_noisy, y):
        X_train, X_test = X_noisy[train_idx], X_noisy[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        acc_scores.append(acc)

    avg_acc = np.mean(acc_scores)
    print(f"Cross-validated Accuracy (predicting tool_classification): {avg_acc:.4f}")
    return clf


# Step 4: Visualize embedding space with PCA
def visualize_pca(df):
    X = np.vstack(df['embedding'].values)
    y = df['text_type'].values

    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X)

    df_pca = pd.DataFrame({
        'PC1': X_reduced[:, 0],
        'PC2': X_reduced[:, 1],
        'Text Type': y,
        'Manipulation Level': df['manipulation_level'].astype(int)
    })

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='Text Type', style='Manipulation Level', palette='Set2')
    plt.title('PCA of Text Embeddings')
    plt.savefig("pca_embedding_plot.png")
    plt.show()

# Step 5: Visualize tool confidence vs. manipulation level
def visualize_confidence(df):
    df['manipulation_level'] = df['manipulation_level'].astype(int)
    df_grouped = df.groupby('manipulation_level')['confidence_score'].mean().reset_index()

    plt.figure(figsize=(8, 5))
    sns.lineplot(data=df_grouped, x='manipulation_level', y='confidence_score', marker='o')
    plt.title("Tool Confidence vs. Manipulation Level")
    plt.xlabel("Manipulation Level")
    plt.ylabel("Average Confidence Score")
    plt.xticks([0, 3, 6, 9])
    plt.savefig("confidence_vs_manipulation.png")
    plt.show()

# Main function to run the pipeline
def main():
    filepath = '440dataset.xlsx'
    df = load_and_clean_data(filepath)
    df = compute_embeddings(df)
    clf = train_classifier(df)
    visualize_pca(df)
    visualize_confidence(df)
    df.to_csv("processed_with_embeddings.csv", index=False)

if __name__ == '__main__':
    main()
