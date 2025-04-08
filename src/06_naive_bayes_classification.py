import pandas as pd
import numpy as np
import os
import re
import nltk
import time
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer # To potentially handle dense conversion

# --- Configuration ---
TEXT_COLUMN = 'text'
CATEGORY_COLUMN = 'category'
SEED = 42
N_SPLITS = 5 # 5-fold stratified cross-validation as per guion.md

# Define paths relative to the src directory
BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, '..', 'data', 'news_reducido.csv')
RESULTS_PATH = os.path.join(BASE_DIR, '..', 'results')

os.makedirs(RESULTS_PATH, exist_ok=True)

# --- NLTK Setup ---
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# --- Custom Tokenizer Function ---
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

# --- Dense Transformer --- #
# Helper to convert sparse matrix to dense for GaussianNB if needed
# Caution: This can use a lot of memory!
def densify(X):
    return X.toarray()

dense_transformer = FunctionTransformer(densify, accept_sparse=True)

# --- Main Naive Bayes Execution ---
if __name__ == "__main__":
    print("--- Starting Naive Bayes Classification Experiment ---")

    # 1. Load Data
    print(f"\n[1] Loading data from {DATA_PATH}...")
    try:
        data_df = pd.read_csv(DATA_PATH)
        X_text = data_df[TEXT_COLUMN].fillna('')
        y_categories = data_df[CATEGORY_COLUMN]
        print(f"Data loaded successfully. {len(X_text)} documents.")
    except Exception as e:
        print(f"An error occurred during data loading: {e}")
        exit()

    # 2. Define Representations (Vectorizers)
    vectorizers = {
        'Binary': CountVectorizer(tokenizer=stemming_tokenizer, binary=True, token_pattern=None),
        'Frequency': CountVectorizer(tokenizer=stemming_tokenizer, token_pattern=None),
        'TF-IDF': TfidfVectorizer(tokenizer=stemming_tokenizer, token_pattern=None)
    }

    # 3. Setup Cross-Validation
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    print(f"\n[2] Using {N_SPLITS}-Fold Stratified Cross-Validation.")

    # --- Naive Bayes Classification --- #
    print("\n[3] Running Naive Bayes Classification (MultinomialNB, GaussianNB)...")
    nb_results = {}

    for rep_name, vectorizer in vectorizers.items():
        print(f"\n--- Testing Naive Bayes with {rep_name} representation ---")
        nb_results[rep_name] = {}
        overall_start_time = time.time()

        # --- MultinomialNB --- #
        print(f"  Testing MultinomialNB...")
        start_time = time.time()
        pipeline_mnb = Pipeline([
            ('vectorizer', vectorizer),
            ('classifier', MultinomialNB())
        ])
        try:
            # Check if representation is suitable (non-negative features)
            # Binary might be problematic conceptually, but technically runs.
            # Run using 2 cores to balance speed and memory
            scores_mnb = cross_val_score(pipeline_mnb, X_text, y_categories, cv=skf, scoring='accuracy', n_jobs=2)
            mean_accuracy_mnb = np.mean(scores_mnb)
            nb_results[rep_name]['MultinomialNB'] = {'accuracy': mean_accuracy_mnb}
            print(f"    MultinomialNB Accuracy: {mean_accuracy_mnb:.4f}")
        except ValueError as ve:
             if "negative values" in str(ve):
                 print("    MultinomialNB Error: Cannot run on data with negative values (likely TF-IDF issue if not handled).")
                 nb_results[rep_name]['MultinomialNB'] = {'accuracy': -999, 'error': 'Negative values'}
             else:
                 print(f"    MultinomialNB Error: {ve}")
                 nb_results[rep_name]['MultinomialNB'] = {'accuracy': -999, 'error': f'ValueError: {ve}'}
        except Exception as e:
            print(f"    MultinomialNB Error: {e}")
            nb_results[rep_name]['MultinomialNB'] = {'accuracy': -999, 'error': f'{type(e).__name__}: {e}'}
        end_time = time.time()
        print(f"    Time taken (MultinomialNB): {end_time - start_time:.2f} seconds")

        # --- GaussianNB --- #
        print(f"  Testing GaussianNB...")
        start_time = time.time()
        # GaussianNB requires dense data. We add a densifier step.
        # WARNING: This step can consume a very large amount of memory!
        pipeline_gnb = Pipeline([
            ('vectorizer', vectorizer),
            ('densifier', dense_transformer), # Convert sparse to dense
            ('classifier', GaussianNB())
        ])
        try:
            # Run using 2 cores to balance speed and memory
            scores_gnb = cross_val_score(pipeline_gnb, X_text, y_categories, cv=skf, scoring='accuracy', n_jobs=2)
            mean_accuracy_gnb = np.mean(scores_gnb)
            nb_results[rep_name]['GaussianNB'] = {'accuracy': mean_accuracy_gnb}
            print(f"    GaussianNB Accuracy: {mean_accuracy_gnb:.4f}")
        except MemoryError:
            print("    GaussianNB Error: MemoryError during densification or fitting. Skipping.")
            nb_results[rep_name]['GaussianNB'] = {'accuracy': -999, 'error': 'MemoryError'}
        except Exception as e:
            print(f"    GaussianNB Error: {e}")
            nb_results[rep_name]['GaussianNB'] = {'accuracy': -999, 'error': f'{type(e).__name__}: {e}'}
        end_time = time.time()
        print(f"    Time taken (GaussianNB): {end_time - start_time:.2f} seconds")

        overall_end_time = time.time()
        print(f"  Total time for {rep_name}: {overall_end_time - overall_start_time:.2f} seconds")

    # --- Summary --- #
    print("\n[4] Naive Bayes Classification Summary:")
    for rep_name, results in nb_results.items():
        print(f"\n--- Representation: {rep_name} ---")
        mnb_acc = results.get('MultinomialNB', {}).get('accuracy', 'N/A')
        gnb_acc = results.get('GaussianNB', {}).get('accuracy', 'N/A')
        mnb_err = results.get('MultinomialNB', {}).get('error')
        gnb_err = results.get('GaussianNB', {}).get('error')

        if mnb_err:
             print(f"  MultinomialNB: Error ({mnb_err})")
        else:
             print(f"  MultinomialNB Accuracy: {mnb_acc:.4f}")

        if gnb_err:
             print(f"  GaussianNB: Error ({gnb_err})")
        else:
             print(f"  GaussianNB Accuracy: {gnb_acc:.4f}")

    print("\n--- Naive Bayes Experiment Finished ---") 