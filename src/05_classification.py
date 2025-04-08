import pandas as pd
import numpy as np
import os
import re
import nltk
import time
import joblib
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# --- Configuration ---
TEXT_COLUMN = 'text'
CATEGORY_COLUMN = 'category'
SEED = 42
N_SPLITS = 5

# Define paths relative to the src directory
BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, '..', 'data', 'news_reducido.csv')
RESULTS_PATH = os.path.join(BASE_DIR, '..', 'results')
MODEL_SAVE_PATH = os.path.join(RESULTS_PATH, 'models')

# Create results & model directories if they don't exist
os.makedirs(RESULTS_PATH, exist_ok=True)
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

# --- NLTK Setup (Copied from previous scripts) ---
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# --- Custom Tokenizer Function (Copied from previous scripts) ---
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

# --- Main Classification Execution ---
if __name__ == "__main__":
    print("--- Starting Classification Experiment (k-NN) ---")

    # 1. Load Data
    print(f"\n[1] Loading data from {DATA_PATH}...")
    try:
        data_df = pd.read_csv(DATA_PATH)
        X_text = data_df[TEXT_COLUMN].fillna('')
        y_categories = data_df[CATEGORY_COLUMN]
        print(f"Data loaded successfully. {len(X_text)} documents.")

        # Encode labels
        print("Encoding string labels to integers...")
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y_categories)
        print(f"Labels encoded. Example: {y_categories.iloc[0]} -> {y_encoded[0]}")
        # Save the label encoder for later use if needed (optional but good practice)
        le_save_path = os.path.join(MODEL_SAVE_PATH, 'label_encoder.joblib')
        joblib.dump(label_encoder, le_save_path)
        print(f"Label encoder saved to {le_save_path}")

    except Exception as e:
        print(f"An error occurred during data loading or label encoding: {e}")
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

    # --- k-NN Classification --- #
    print("\n[3] Running k-Nearest Neighbors (k-NN) Classification with GridSearchCV...")
    knn_results = {}

    knn_param_grid = {
        'classifier__n_neighbors': [3, 5, 7, 9],
        'classifier__weights': ['uniform', 'distance'],
        'classifier__p': [1, 2]
    }

    for rep_name, vectorizer in vectorizers.items():
        print(f"\n--- Testing k-NN with {rep_name} representation ---")
        start_time = time.time()

        pipeline = Pipeline([
            ('vectorizer', vectorizer),
            ('classifier', KNeighborsClassifier())
        ])

        grid_search = GridSearchCV(pipeline, knn_param_grid, cv=skf, scoring='accuracy', n_jobs=-1, verbose=1)

        try:
            # Fit using ENCODED labels
            grid_search.fit(X_text, y_encoded)

            best_score = grid_search.best_score_
            best_params = grid_search.best_params_
            best_estimator = grid_search.best_estimator_ # The final fitted pipeline

            knn_results[rep_name] = {
                'best_accuracy': best_score,
                'best_params': best_params
            }

            print(f"  Best Accuracy ({rep_name}): {best_score:.4f}")
            print(f"  Best Parameters ({rep_name}): {best_params}")

            # Save the best estimator (pipeline)
            model_filename = os.path.join(MODEL_SAVE_PATH, f'knn_best_{rep_name}.joblib')
            joblib.dump(best_estimator, model_filename)
            print(f"  Best k-NN model for {rep_name} saved to: {model_filename}")

        except Exception as e:
            print(f"  Error during k-NN Grid Search for {rep_name}: {e}")
            knn_results[rep_name] = {'best_accuracy': -999, 'best_params': f"Error ({type(e).__name__})"}
            print(f"  Failed to save model for {rep_name} due to error.")

        end_time = time.time()
        print(f"  Time taken for {rep_name}: {end_time - start_time:.2f} seconds")

    # --- Naive Bayes Classification (Placeholder) --- #
    print("\n[4] Skipping Naive Bayes Classification (run src/06_naive_bayes_classification.py separately)...")

    # --- Final Comparison (Partial) --- #
    print("\n[5] Classification Experiment Summary (k-NN Only):")
    print("\n--- k-NN Results ---")
    for rep_name, result in knn_results.items():
         # Format parameters for printing
         params_str = ', '.join([f'{k.split("__")[1]}={v}' for k,v in result.get('best_params', {}).items()])
         print(f"  {rep_name}: Best Accuracy = {result.get('best_accuracy', 'N/A'):.4f}, Params = {{{params_str}}}")

    print("\n--- k-NN Classification Experiment Finished ---") 