import pandas as pd
import numpy as np
import re
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer # Added for Stemming
# from nltk.stem import WordNetLemmatizer # Kept commented

# --- Configuration ---
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'news_reducido.csv')
TEXT_COLUMN = 'text' # Column to preprocess
CLEANED_TEXT_COLUMN = 'cleaned_text' # Intermediate column
# TOKENIZED_TEXT_COLUMN = 'tokens' # Intermediate column
FINAL_TOKENS_COLUMN = 'processed_tokens' # Final preprocessed tokens column (stemmed)
CATEGORY_COLUMN = 'category' # Target column
SEED = 42 # For reproducibility

# --- NLTK Setup ---
# Downloads should already be done from previous steps
# Load stopwords once
stop_words = set(stopwords.words('english'))
# Instantiate Stemmer
stemmer = PorterStemmer()

# --- Text Cleaning Functions ---

def clean_text(text):
    """Applies basic text cleaning steps."""
    if not isinstance(text, str):
        return "" # Handle non-string inputs, including NaN after fillna

    # 1. Convert to lowercase
    text = text.lower()

    # 2. Remove punctuation and numbers
    text = re.sub(r'[^a-z\s]', '', text)

    # 3. Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# --- Tokenization, Stopword Removal, and Stemming ---

def process_tokens(text):
    """Tokenizes, removes stopwords, and applies stemming."""
    # 1. Tokenize
    tokens = word_tokenize(text)
    # 2. Remove stopwords
    filtered_tokens = [word for word in tokens if word not in stop_words]
    # 3. Apply stemming
    stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
    return stemmed_tokens

# --- Main Preprocessing Function (Updated) ---

def preprocess_data(df):
    """Loads data and applies cleaning, tokenization, stopword removal, and stemming."""
    print("--- Starting Preprocessing --- C:\\Users\\jfdg0\\Documents\\Asignaturas\\Minería Web\\mw-2>")

    # 1. Handle Missing Values in text column
    print(f"Handling missing values in '{TEXT_COLUMN}'...")
    original_missing = df[TEXT_COLUMN].isnull().sum()
    df[TEXT_COLUMN] = df[TEXT_COLUMN].fillna('')
    print(f"Replaced {original_missing} missing values with empty strings.")

    # 2. Apply Text Cleaning
    print(f"Applying text cleaning to '{TEXT_COLUMN}'...")
    df[CLEANED_TEXT_COLUMN] = df[TEXT_COLUMN].apply(clean_text)
    print(f"Created '{CLEANED_TEXT_COLUMN}' column.")

    # 3. Tokenize, Remove Stopwords, and Stem
    print(f"Processing tokens (tokenize, stopwords, stem) from '{CLEANED_TEXT_COLUMN}'...")
    # This step can take some time
    df[FINAL_TOKENS_COLUMN] = df[CLEANED_TEXT_COLUMN].apply(process_tokens)
    print(f"Created '{FINAL_TOKENS_COLUMN}' column.")

    # Display comparison for a sample
    print("\nSample Text Comparison (Original vs Cleaned vs Processed Tokens):")
    sample_indices = [0, 1, 2] # Show first few rows
    for i in sample_indices:
        if i < len(df):
            print(f"--- Example {i+1} ---")
            print(f"Original:    {df[TEXT_COLUMN].iloc[i][:150]}...")
            print(f"Cleaned:     {df[CLEANED_TEXT_COLUMN].iloc[i][:150]}...")
            print(f"Tokens:      {df[FINAL_TOKENS_COLUMN].iloc[i][:20]}...") # Show first 20 processed tokens
        else:
            print(f"Index {i} out of bounds for sample display.")

    print("\n--- Preprocessing (Cleaning, Tokenization, Stopwords, Stemming) Finished --- C:\\Users\\jfdg0\\Documents\\Asignaturas\\Minería Web\\mw-2>")
    # Return only the essential columns for further steps
    return df[[CATEGORY_COLUMN, FINAL_TOKENS_COLUMN]]

# --- Execution Example ---
if __name__ == "__main__":
    print(f"Loading data from {DATA_PATH}...")
    try:
        data_df = pd.read_csv(DATA_PATH)
        print("Data loaded successfully.")
        # Keep only necessary columns for preprocessing input
        data_df_subset = data_df[[TEXT_COLUMN, CATEGORY_COLUMN]].copy()
        processed_df = preprocess_data(data_df_subset)
        print(f"\nProcessed DataFrame shape: {processed_df.shape}")
        print("Processed DataFrame head:")
        print(processed_df.head())

        # Optional: Save the processed data
        # save_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'news_reducido_processed.pkl')
        # processed_df.to_pickle(save_path)
        # print(f"Processed data saved to {save_path}")

    except FileNotFoundError:
        print(f"Error: Dataset file not found at {DATA_PATH}")
    except Exception as e:
        print(f"An error occurred: {e}") 