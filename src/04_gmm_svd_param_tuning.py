import pandas as pd
import numpy as np
import os
import re
import nltk
import time
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

from sklearn.mixture import GaussianMixture
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.metrics import adjusted_rand_score, homogeneity_completeness_v_measure
from sklearn.feature_extraction.text import TfidfVectorizer

# --- Configuration ---
TEXT_COLUMN = 'text'
CATEGORY_COLUMN = 'category'
SEED = 42
GMM_COMPONENTS = 4 # Fixed according to guion.md

# Define paths relative to the src directory
BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, '..', 'data', 'news_reducido.csv')
RESULTS_PATH = os.path.join(BASE_DIR, '..', 'results') # Although we might not save specific files here

# SVD Components to test
SVD_COMPONENTS_TO_TEST = [50, 100, 150, 200, 250] # Example values

# --- NLTK Setup ---
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# --- Custom Tokenizer Function (Copied from 03_kmeans_clustering.py) ---
def clean_text_minimal(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def stemming_tokenizer(text):
    cleaned_text = clean_text_minimal(text)
    tokens = word_tokenize(cleaned_text)
    processed_tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return processed_tokens

# --- Vectorizer Creation Function (TF-IDF only) ---
def create_tfidf_vectorizer():
    return TfidfVectorizer(tokenizer=stemming_tokenizer)

# --- Evaluation Function (Copied from 03_kmeans_clustering.py, simplified print) ---
def evaluate_clustering_simple(X_vec_eval, labels_pred, labels_true, svd_n, gmm_n):
    ari = adjusted_rand_score(labels_true, labels_pred)
    hom, com, v_measure = homogeneity_completeness_v_measure(labels_true, labels_pred)
    print(f"    - ARI: {ari:.4f}, V-measure: {v_measure:.4f} (Hom: {hom:.4f}, Com: {com:.4f})")
    # Calculate internal metrics on the SVD space for reference (not primary selection criteria)
    try:
        sil = silhouette_score(X_vec_eval, labels_pred)
        print(f"    - Silhouette (SVD space): {sil:.4f}")
    except Exception: sil = -999
    try:
        db = davies_bouldin_score(X_vec_eval, labels_pred)
        print(f"    - Davies-Bouldin (SVD space): {db:.4f}")
    except Exception: db = -999
    return {'ari': ari, 'v_measure': v_measure, 'homogeneity': hom, 'completeness': com, 'silhouette_svd': sil, 'db_svd': db}

# --- Main Tuning Execution ---
if __name__ == "__main__":
    print("--- Starting SVD n_components Tuning for GMM (TF-IDF) ---")

    # 1. Load Data
    print(f"\n[1] Loading data from {DATA_PATH}...")
    try:
        data_df = pd.read_csv(DATA_PATH)
        X_text = data_df[TEXT_COLUMN].fillna('')
        y_categories = data_df[CATEGORY_COLUMN]
        le = LabelEncoder()
        y_true = le.fit_transform(y_categories)
        true_category_names = le.classes_
        print(f"Data loaded successfully. {len(X_text)} documents.")
    except Exception as e:
        print(f"An error occurred during data loading: {e}")
        exit()

    # 2. Vectorize with TF-IDF
    print("\n[2] Vectorizing data with TF-IDF...")
    try:
        tfidf_vectorizer = create_tfidf_vectorizer()
        X_tfidf_vec = tfidf_vectorizer.fit_transform(X_text)
        print(f"TF-IDF Vectorization complete. Shape: {X_tfidf_vec.shape}")
    except Exception as e:
        print(f"Error during TF-IDF vectorization: {e}")
        exit()

    # 3. Tune SVD n_components
    print(f"\n[3] Testing TruncatedSVD n_components: {SVD_COMPONENTS_TO_TEST} with GMM (n={GMM_COMPONENTS})...")
    tuning_results = {}
    best_ari = -1
    best_vmeasure = -1
    best_svd_n_ari = -1
    best_svd_n_vmeasure = -1

    for n_svd in SVD_COMPONENTS_TO_TEST:
        print(f"\n--- Testing SVD n_components = {n_svd} ---")
        start_time = time.time()
        try:
            # Apply SVD
            print(f"  Applying TruncatedSVD...")
            svd = TruncatedSVD(n_components=n_svd, random_state=SEED)
            X_tfidf_svd = svd.fit_transform(X_tfidf_vec)
            print(f"  Dimensionality reduced to: {X_tfidf_svd.shape}")

            # Fit GMM
            print(f"  Fitting GMM (n={GMM_COMPONENTS})...")
            gmm = GaussianMixture(n_components=GMM_COMPONENTS, random_state=SEED, covariance_type='full')
            gmm.fit(X_tfidf_svd)
            labels_gmm = gmm.predict(X_tfidf_svd)

            # Evaluate
            print(f"  Evaluating...")
            metrics = evaluate_clustering_simple(X_tfidf_svd, labels_gmm, y_true, n_svd, GMM_COMPONENTS)
            tuning_results[n_svd] = metrics

            # Track best based on external metrics
            if metrics['ari'] > best_ari:
                best_ari = metrics['ari']
                best_svd_n_ari = n_svd
            if metrics['v_measure'] > best_vmeasure:
                best_vmeasure = metrics['v_measure']
                best_svd_n_vmeasure = n_svd

        except MemoryError:
            print(f"  Error: MemoryError during SVD/GMM for n_components={n_svd}. Skipping.")
            tuning_results[n_svd] = "Error (Memory)"
        except Exception as e:
            print(f"  Error during processing for n_components={n_svd}: {e}")
            tuning_results[n_svd] = f"Error ({type(e).__name__})"

        end_time = time.time()
        print(f"  Time taken: {end_time - start_time:.2f} seconds")

    # 4. Summary of Tuning
    print("\n--- Tuning Summary ---")
    print("Results per SVD n_components:")
    for n_svd, metrics in tuning_results.items():
        if isinstance(metrics, dict):
             print(f"  SVD={n_svd}: ARI={metrics['ari']:.4f}, V-measure={metrics['v_measure']:.4f}")
        else:
             print(f"  SVD={n_svd}: {metrics}") # Print error message

    print(f"\nBest SVD n_components based on ARI: {best_svd_n_ari} (ARI = {best_ari:.4f})")
    print(f"Best SVD n_components based on V-measure: {best_svd_n_vmeasure} (V-measure = {best_vmeasure:.4f})")

    # Choose final SVD n based on consensus or priority (e.g., ARI)
    final_best_svd_n = best_svd_n_ari if best_svd_n_ari == best_svd_n_vmeasure else best_svd_n_ari # Prioritize ARI if different
    print(f"\n--> Recommended SVD n_components for main script: {final_best_svd_n}")

    print("\n--- Tuning Script Finished ---") 