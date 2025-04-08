import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define paths relative to the script directory
# Go one level up from src (..) and then into data
DATA_PATH = os.path.join(script_dir, '..', 'data', 'news_reducido.csv')
# Go one level up from src (..) and then into results
RESULTS_PATH = os.path.join(script_dir, '..', 'results')

# Ensure results directory exists
os.makedirs(RESULTS_PATH, exist_ok=True)

# Set plotting style
sns.set_theme(style="whitegrid")

def run_eda():
    """Performs Exploratory Data Analysis on the reduced dataset."""
    print("--- Starting Exploratory Data Analysis ---")

    # --- 1. Load Data ---
    print(f"\n[1] Loading dataset from: {DATA_PATH}")
    try:
        df = pd.read_csv(DATA_PATH)
        print("Dataset loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {DATA_PATH}")
        return # Exit if data can't be loaded

    # --- 2. Initial Data Inspection ---
    print("\n[2] Initial Data Inspection:")
    print("  \nDataset Info:")
    df.info()
    print(f"\n  Dataset Shape: {df.shape}")
    print("\n  First 5 rows:")
    print(df.head())

    # --- 3. Missing Values Analysis ---
    print("\n[3] Missing Values Analysis:")
    missing_values = df.isnull().sum()
    print("  Missing values per column:")
    print(missing_values[missing_values > 0]) # Only show columns with missing values
    if missing_values.sum() == 0:
        print("  No missing values found.")

    # --- 4. Category Analysis ---
    print("\n[4] Category Analysis:")
    if 'category' in df.columns:
        unique_categories = df['category'].unique()
        print(f"  Unique Categories ({len(unique_categories)}): {unique_categories}")

        print("\n  Category Distribution:")
        category_counts = df['category'].value_counts()
        print(category_counts)

        # Plot distribution and save
        plt.figure(figsize=(10, 6))
        sns.barplot(x=category_counts.index, y=category_counts.values)
        plt.title('Distribution of Categories')
        plt.xlabel('Category')
        plt.ylabel('Number of Articles')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plot_path = os.path.join(RESULTS_PATH, 'category_distribution.png')
        plt.savefig(plot_path)
        print(f"  Category distribution plot saved to: {plot_path}")
        plt.close() # Close the plot to free memory
    else:
        print("  'category' column not found.")

    # --- 5. Text Field Analysis ---
    print("\n[5] Text Field Length Analysis:")
    text_fields = ['headline', 'text', 'short_description']
    length_stats = {}

    if not df.empty:
        # Calculate lengths
        for field in text_fields:
            if field in df.columns:
                # Ensure the column is string type and handle potential NaNs
                df[f'{field}_len'] = df[field].fillna('').astype(str).apply(len)
                length_stats[field] = df[f'{field}_len'].describe()
                print(f"\n  Statistics for '{field}' length:")
                print(length_stats[field])
            else:
                print(f"  Column '{field}' not found.")

        # Plot histograms for text lengths and save
        num_fields_found = sum(1 for field in text_fields if f'{field}_len' in df.columns)
        if num_fields_found > 0:
            plt.figure(figsize=(6 * num_fields_found, 5))
            plot_idx = 1
            for field in text_fields:
                len_col = f'{field}_len'
                if len_col in df.columns:
                    plt.subplot(1, num_fields_found, plot_idx)
                    sns.histplot(df[len_col], bins=50, kde=False) # KDE can be slow for large text
                    plt.title(f'Distribution of {field.capitalize()} Length')
                    plt.xlabel('Length (characters)')
                    plt.ylabel('Frequency')
                    plot_idx += 1

            plt.tight_layout()
            plot_path = os.path.join(RESULTS_PATH, 'text_length_distributions.png')
            plt.savefig(plot_path)
            print(f"\n  Text length distribution plots saved to: {plot_path}")
            plt.close()
    else:
        print("  DataFrame is empty, cannot analyze text lengths.")


    # --- 6. Decision on Text Field(s) ---
    print("\n[6] Decision on Text Field(s) for Analysis:")
    print("  - Based on the guidance in guion.md and the substantial content observed,")
    print("    the primary field for analysis will be: 'text'.")
    print("  - Rationale: Contains the most detailed information for clustering/classification.")

    print("\n--- EDA Finished ---")

if __name__ == "__main__":
    run_eda() 