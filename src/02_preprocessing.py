import pandas as pd
import numpy as np
import re
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# --- Configuration ---
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'news_reducido.csv')
TEXT_COLUMN = 'text'
CATEGORY_COLUMN = 'category'
SEED = 42

# --- NLTK Setup ---
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# --- Custom Tokenizer Function ---

def clean_text_minimal(text):
    """Minimal cleaning: lowercase and remove non-alphanumeric chars (keep spaces)."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text) # Keep letters and spaces
    text = re.sub(r'\s+', ' ', text).strip() # Consolidate whitespace
    return text

def stemming_tokenizer(text):
    """Cleans, tokenizes, removes stopwords, and stems."""
    cleaned_text = clean_text_minimal(text)
    tokens = word_tokenize(cleaned_text)
    processed_tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    # Return the list of tokens, scikit-learn vectorizers handle joining if needed internally
    # but passing the list directly is usually fine and expected by tokenizer param
    return processed_tokens

# --- Pipeline Creation Functions ---

def create_binary_pipeline():
    """Creates a pipeline for Binary representation."""
    return Pipeline([
        ('vectorizer', CountVectorizer(tokenizer=stemming_tokenizer, binary=True))
    ])

def create_frequency_pipeline():
    """Creates a pipeline for Frequency (Count) representation."""
    return Pipeline([
        ('vectorizer', CountVectorizer(tokenizer=stemming_tokenizer))
    ])

def create_tfidf_pipeline():
    """Creates a pipeline for TF-IDF representation."""
    return Pipeline([
        ('vectorizer', TfidfVectorizer(tokenizer=stemming_tokenizer))
    ])

# --- Execution Example --- (Demonstrates using one pipeline)
if __name__ == "__main__":
    print(f"Loading data from {DATA_PATH}...")
    try:
        data_df = pd.read_csv(DATA_PATH)
        print("Data loaded successfully.")

        # Separate features (text data) and target (categories)
        X = data_df[TEXT_COLUMN].fillna('') # Ensure NaNs are handled before pipeline
        y = data_df[CATEGORY_COLUMN]

        print(f"\n--- Demonstrating TF-IDF Pipeline --- C:\\Users\\jfdg0\\Documents\\Asignaturas\\Minería Web\\mw-2>")

        # Create the TF-IDF pipeline
        tfidf_pipeline = create_tfidf_pipeline()

        # Fit the pipeline to the data and transform it
        # NOTE: fit_transform should ideally be done only on training data in a real scenario
        # Here, we apply it to all data for demonstration of the pipeline structure.
        print("Fitting and transforming data with TF-IDF pipeline...")
        X_tfidf = tfidf_pipeline.fit_transform(X)

        print("\nTransformation complete.")
        print(f"Shape of TF-IDF matrix: {X_tfidf.shape}")
        print(f"Type of TF-IDF matrix: {type(X_tfidf)}")

        # Show feature names (vocabulary) - may be large
        try:
            feature_names = tfidf_pipeline.named_steps['vectorizer'].get_feature_names_out()
            print(f"Vocabulary size: {len(feature_names)}")
            print(f"Sample vocabulary: {feature_names[:10]}... {feature_names[-10:]}")
        except Exception as e:
            print(f"Could not retrieve feature names: {e}")

        print("\n--- Pipeline demonstration finished --- C:\\Users\\jfdg0\\Documents\\Asignaturas\\Minería Web\\mw-2>")

    except FileNotFoundError:
        print(f"Error: Dataset file not found at {DATA_PATH}")
    except Exception as e:
        print(f"An error occurred: {e}") 