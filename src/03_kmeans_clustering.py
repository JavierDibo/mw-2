import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.metrics import adjusted_rand_score, homogeneity_completeness_v_measure
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# --- Configuration copied from preprocessing.py and local ---
TEXT_COLUMN = 'text'
CATEGORY_COLUMN = 'category'
SEED = 42
K = 4 # Número de clusters según guion.md
# Define paths relative to the src directory
BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, '..', 'data', 'news_reducido.csv')
RESULTS_PATH = os.path.join(BASE_DIR, '..', 'results')
ASSIGNMENTS_FILE_TPL = os.path.join(RESULTS_PATH, 'kmeans_assignments_{}.csv')
TSNE_PLOT_FILE_TPL = os.path.join(RESULTS_PATH, 'tsne_kmeans_{}.png')
TSNE_PLOT_TRUE_FILE_TPL = os.path.join(RESULTS_PATH, 'tsne_true_categories_{}.png')

# Semillas para probar la sensibilidad de K-Means
RANDOM_STATES_TO_TEST = [0, SEED, 123, 2024, 999]

# Crear directorio de resultados si no existe
os.makedirs(RESULTS_PATH, exist_ok=True)

sns.set_theme(style="whitegrid")

# --- NLTK Setup copied from preprocessing.py ---
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# --- Custom Tokenizer Function copied from preprocessing.py ---
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
    return processed_tokens

# --- Pipeline Creation Functions copied from preprocessing.py ---
def create_binary_pipeline():
    """Creates a pipeline for Binary representation."""
    # We define the vectorizer directly here now, not a pipeline containing one
    return CountVectorizer(tokenizer=stemming_tokenizer, binary=True)

def create_frequency_pipeline():
    """Creates a pipeline for Frequency (Count) representation."""
    return CountVectorizer(tokenizer=stemming_tokenizer)

def create_tfidf_pipeline():
    """Creates a pipeline for TF-IDF representation."""
    return TfidfVectorizer(tokenizer=stemming_tokenizer)

# --- Helper Functions (evaluate_clustering, visualize_tsne) ---
def evaluate_clustering(X_vec, labels_pred, labels_true, representation_name, seed):
    """Calculates and prints clustering evaluation metrics."""
    print(f"  \n  Evaluation for {representation_name} (seed={seed}):")
    try:
        sil = silhouette_score(X_vec, labels_pred)
        print(f"    - Silhouette Score:       {sil:.4f}")
    except ValueError as e:
        sil = -999 # Error value
        print(f"    - Silhouette Score:       Error ({e})")

    try:
        # Ensure input is dense for Davies-Bouldin if it's sparse
        X_dense = X_vec.toarray() if hasattr(X_vec, "toarray") else X_vec
        # Check for single cluster case which causes error in DB score
        if len(np.unique(labels_pred)) < 2:
            print("    - Davies-Bouldin Score:   Cannot calculate (less than 2 clusters found)")
            db = -999
        else:
            db = davies_bouldin_score(X_dense, labels_pred)
            print(f"    - Davies-Bouldin Score:   {db:.4f}")
    except Exception as e:
        db = -999 # Error value
        print(f"    - Davies-Bouldin Score:   Error ({e})")

    # External (require true labels)
    ari = adjusted_rand_score(labels_true, labels_pred)
    hom, com, v_measure = homogeneity_completeness_v_measure(labels_true, labels_pred)
    print(f"    - Adjusted Rand Index:    {ari:.4f}")
    print(f"    - Homogeneity:            {hom:.4f}")
    print(f"    - Completeness:           {com:.4f}")
    print(f"    - V-measure:              {v_measure:.4f}")
    return {'silhouette': sil, 'davies_bouldin': db, 'ari': ari, 'homogeneity': hom, 'completeness': com, 'v_measure': v_measure}

def visualize_tsne(X_vec, labels_pred, labels_true, representation_name, filename_pred, filename_true):
    """Applies t-SNE and saves scatter plots colored by predicted and true labels."""
    print(f"  \n  Applying t-SNE for {representation_name} visualization (this may take a moment)...")
    # Ensure input is dense for t-SNE if it's sparse
    X_dense = X_vec.toarray() if hasattr(X_vec, "toarray") else X_vec
    tsne = TSNE(n_components=2, random_state=SEED, perplexity=30, n_iter=300, init='pca') # Added init='pca' for stability
    X_tsne = tsne.fit_transform(X_dense)
    print("  t-SNE finished.")

    df_tsne = pd.DataFrame(X_tsne, columns=['tsne1', 'tsne2'])
    df_tsne['predicted_cluster'] = labels_pred
    df_tsne['true_category'] = labels_true # Assumes labels_true are original category names or mapped

    # Plot colored by predicted clusters
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        x="tsne1", y="tsne2",
        hue="predicted_cluster",
        palette=sns.color_palette("hsv", K),
        data=df_tsne,
        legend="full",
        alpha=0.6
    )
    plt.title(f't-SNE Visualization of K-Means Clusters ({representation_name})')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.savefig(filename_pred)
    plt.close()
    print(f"    - Saved t-SNE plot (predicted clusters) to: {filename_pred}")

    # Plot colored by true categories
    plt.figure(figsize=(12, 8))
    unique_cats = df_tsne['true_category'].unique()
    palette_true = sns.color_palette("tab10", len(unique_cats))
    sns.scatterplot(
        x="tsne1", y="tsne2",
        hue="true_category",
        palette=palette_true,
        hue_order=sorted(unique_cats), # Ensure consistent color mapping
        data=df_tsne,
        legend="full",
        alpha=0.6
    )
    plt.title(f't-SNE Visualization by True Category ({representation_name})')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.savefig(filename_true)
    plt.close()
    print(f"    - Saved t-SNE plot (true categories) to: {filename_true}")

# --- Main Execution ---
if __name__ == "__main__":
    print("--- Starting K-Means Clustering Experiment --- C:\\Users\\jfdg0\\Documents\\Asignaturas\\Minería Web\\mw-2\\src>") # Adjusted path indicator

    # 1. Load Data
    print(f"\n[1] Loading data from {DATA_PATH}...")
    try:
        data_df = pd.read_csv(DATA_PATH)
        X_text = data_df[TEXT_COLUMN].fillna('')
        y_categories = data_df[CATEGORY_COLUMN]
        # Encode categories to numeric labels for some metrics
        le = LabelEncoder()
        y_true = le.fit_transform(y_categories)
        true_category_names = le.classes_
        print(f"Data loaded successfully. Found {len(true_category_names)} categories: {true_category_names}")
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {DATA_PATH}")
        exit()
    except Exception as e:
        print(f"An error occurred during data loading: {e}")
        exit()

    # 2. Define Representation Vectorizers (no longer pipelines)
    vectorizers_to_run = {
        'Binary': create_binary_pipeline(),
        'Frequency': create_frequency_pipeline(),
        'TF-IDF': create_tfidf_pipeline()
    }

    all_results = {}

    # 3. Iterate through Representations
    print("\n[2] Running K-Means for each representation...")
    for name, vectorizer in vectorizers_to_run.items():
        print(f"\n--- Processing Representation: {name} ---")
        all_results[name] = {}

        try:
             print(f"  Vectorizing data for {name}...")
             # Apply vectorizer directly
             X_vec = vectorizer.fit_transform(X_text)
             print(f"  Vectorization complete. Shape: {X_vec.shape}")
        except Exception as e:
             print(f"    Error during vectorization for {name}: {e}")
             continue # Skip to next representation if vectorization fails

        # Create KMeans instance
        kmeans = KMeans(n_clusters=K, n_init='auto')

        # a) Test Sensitivity to Random States
        print(f"  \n  a) Testing sensitivity to random_state for KMeans ({name})...")
        sensitivity_results = {}
        for state in RANDOM_STATES_TO_TEST:
            print(f"    Running with random_state={state}...")
            kmeans.set_params(random_state=state)
            try:
                # Fit KMeans on the vectorized data
                kmeans.fit(X_vec)
                labels = kmeans.labels_
                # Evaluate
                metrics = evaluate_clustering(X_vec, labels, y_true, name, state)
                sensitivity_results[state] = metrics
            except Exception as e:
                print(f"      Error during KMeans fit/evaluation for seed {state}: {e}")
                sensitivity_results[state] = None # Mark as errored

        # Discuss sensitivity (simple print)
        print("  \n  Sensitivity Analysis Summary:")
        aris = {s: r['ari'] for s, r in sensitivity_results.items() if r}
        if aris:
             print(f"    Adjusted Rand Index varied across seeds: {aris}")
             print("    -> K-Means results depend on initial centroid positions.")
        else:
             print("    Could not perform sensitivity analysis due to errors.")

        # b) Best Run (using fixed SEED for reproducibility)
        print(f"  \n  b) Performing main run with fixed seed ({SEED})...")
        kmeans.set_params(random_state=SEED)
        try:
            print(f"    Fitting final KMeans for {name}...")
            kmeans.fit(X_vec) # Fit on already vectorized data
            labels_best = kmeans.labels_
        except Exception as e:
            print(f"    Error during final KMeans fit for {name} with seed {SEED}: {e}")
            continue # Skip to next representation if final run fails

        # c) Evaluate Best Run
        print(f"  \n  c) Evaluating best run (seed={SEED})...")
        best_metrics = evaluate_clustering(X_vec, labels_best, y_true, name, f"SEED={SEED}")
        all_results[name]['metrics'] = best_metrics

        # d) Save Assignments
        print(f"  \n  d) Saving cluster assignments...")
        assignments_df = pd.DataFrame({
            'document_index': X_text.index,
            'true_category': y_categories, # Original category names
            'kmeans_cluster': labels_best
        })
        assignments_file = ASSIGNMENTS_FILE_TPL.format(name)
        try:
            assignments_df.to_csv(assignments_file, index=False)
            print(f"    - Assignments saved to: {assignments_file}")
            all_results[name]['assignments_file'] = assignments_file
        except Exception as e:
            print(f"    - Error saving assignments: {e}")

        # e) Visualize with t-SNE
        print(f"  \n  e) Generating t-SNE visualizations...")
        tsne_file = TSNE_PLOT_FILE_TPL.format(name)
        tsne_true_file = TSNE_PLOT_TRUE_FILE_TPL.format(name)
        try:
            # Use original category names for the true label plot
            visualize_tsne(X_vec, labels_best, y_categories, name, tsne_file, tsne_true_file)
            all_results[name]['tsne_plot_pred'] = tsne_file
            all_results[name]['tsne_plot_true'] = tsne_true_file
        except Exception as e:
            print(f"    - Error during t-SNE visualization: {e}")

    # 4. Final Summary (Basic)
    print("\n[3] K-Means Experiment Summary:")
    for name, results_dict in all_results.items():
        print(f"\n--- Results for: {name} ---")
        if 'metrics' in results_dict:
            print("  Metrics (Seed=42):")
            for metric, value in results_dict['metrics'].items():
                # Handle potential error values
                value_str = f"{value:.4f}" if isinstance(value, (int, float)) and value != -999 else str(value)
                print(f"    - {metric.replace('_', ' ').capitalize()}: {value_str}")
        else:
             print("  Metrics: Not available due to errors.")
        if 'assignments_file' in results_dict:
             print(f"  Assignments File: {results_dict['assignments_file']}")
        if 'tsne_plot_pred' in results_dict:
             print(f"  t-SNE Plots: {results_dict['tsne_plot_pred']} (predicted), {results_dict['tsne_plot_true']} (true)")

    print(f"\n--- K-Means Clustering Experiment Finished --- C:\\Users\\jfdg0\\Documents\\Asignaturas\\Minería Web\\mw-2\\src>") # Adjusted path indicator 