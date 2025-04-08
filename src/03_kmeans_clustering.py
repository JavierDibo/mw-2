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
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import TruncatedSVD

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
# Add filenames for GMM results
GMM_ASSIGNMENTS_FILE_TPL = os.path.join(RESULTS_PATH, 'gmm_assignments_{}.csv') # Optional, but good practice
SVD_N_COMPONENTS = 100 # Number of components for TruncatedSVD

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
    print("--- Starting Clustering Experiment ---") # Simplified title

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

        # Discuss sensitivity
        print("  \n  Sensitivity Analysis Summary:")
        aris = {s: r['ari'] for s, r in sensitivity_results.items() if r}
        if aris:
             print(f"    Adjusted Rand Index varied across seeds: {aris}")
             # Explanation for K-Means Sensitivity
             print("    -> Theoretical Explanation: K-Means is an iterative algorithm that aims to minimize the within-cluster sum of squares.")
             print("       It starts with an initial guess for the cluster centroids (either randomly or using a strategy like k-means++).")
             print("       The algorithm converges to a local minimum, which depends heavily on the starting positions.")
             print("       Different initializations can lead to different local minima and thus different final cluster assignments and performance.")
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
        try:
            assignments_df = pd.DataFrame({'document_index': X_text.index, 'cluster': labels_best})
            assignments_file = ASSIGNMENTS_FILE_TPL.format(name)
            assignments_df.to_csv(assignments_file, index=False)
            print(f"    - Assignments saved to: {assignments_file}")
            all_results[name]['assignments_file'] = assignments_file
        except Exception as e:
            print(f"    Error saving assignments for {name}: {e}")
            all_results[name]['assignments_file'] = "Error"

        # e) Generate Visualizations
        print(f"  \n  e) Generating t-SNE visualizations...")
        try:
            tsne_pred_file = TSNE_PLOT_FILE_TPL.format(name)
            tsne_true_file = TSNE_PLOT_TRUE_FILE_TPL.format(name)
            # Check if data is too large or likely to cause memory issues for t-SNE visualization
            if X_vec.shape[0] * X_vec.shape[1] > 1e8: # Heuristic limit
                 print("    Skipping t-SNE visualization due to potentially large data size.")
                 all_results[name]['tsne_plots'] = "Skipped (Large Data)"
            else:
                 visualize_tsne(X_vec, labels_best, y_categories, name, tsne_pred_file, tsne_true_file)
                 all_results[name]['tsne_plots'] = (tsne_pred_file, tsne_true_file)
        except MemoryError:
             print("    Error: MemoryError during t-SNE calculation. Skipping visualization.")
             all_results[name]['tsne_plots'] = "Error (Memory)"
        except Exception as e:
             print(f"    Error during t-SNE visualization for {name}: {e}")
             all_results[name]['tsne_plots'] = f"Error ({type(e).__name__})"

    # --- Section 4: Gaussian Mixture Model (Only for TF-IDF with SVD) ---
    print(f"\n[3] Running Gaussian Mixture Model (GMM) for TF-IDF representation with TruncatedSVD (n_components={SVD_N_COMPONENTS})...")
    gmm_results = {}
    if 'TF-IDF' in vectorizers_to_run:
        tfidf_vectorizer = vectorizers_to_run['TF-IDF']
        try:
            # Re-vectorize or reuse TF-IDF data
            if 'TF-IDF' not in all_results or 'metrics' not in all_results['TF-IDF']:
                 print("  TF-IDF K-Means results not available, re-vectorizing for GMM...")
                 X_tfidf_vec = tfidf_vectorizer.fit_transform(X_text)
            else:
                 print("  Reusing TF-IDF vectorized data from K-Means step.")
                 X_tfidf_vec = X_vec # Assumes TF-IDF was the last one processed

            # Apply TruncatedSVD
            print(f"  Applying TruncatedSVD with n_components={SVD_N_COMPONENTS}...")
            svd = TruncatedSVD(n_components=SVD_N_COMPONENTS, random_state=SEED)
            X_tfidf_svd = svd.fit_transform(X_tfidf_vec)
            print(f"  Dimensionality reduced to: {X_tfidf_svd.shape}")

            # Fit GMM on reduced data
            print(f"  Fitting GMM with n_components={K} and random_state={SEED} on SVD data...")
            gmm = GaussianMixture(n_components=K, random_state=SEED, covariance_type='full')
            # GMM needs dense data, SVD output is already dense
            gmm.fit(X_tfidf_svd)
            labels_gmm = gmm.predict(X_tfidf_svd)

            # Evaluate GMM
            print(f"  \n  Evaluating GMM run (seed={SEED}) on SVD data...")
            # Note: Internal metrics (Silhouette, DB) are calculated on X_tfidf_svd
            # External metrics (ARI, etc.) use original y_true and are comparable to K-Means
            gmm_metrics = evaluate_clustering(X_tfidf_svd, labels_gmm, y_true, "TF-IDF (SVD + GMM)", f"SEED={SEED}")
            gmm_results = {'metrics': gmm_metrics}

            # Save GMM Assignments
            print(f"  \n  Saving GMM cluster assignments (from SVD data)...")
            try:
                gmm_assignments_df = pd.DataFrame({'document_index': X_text.index, 'cluster': labels_gmm})
                # Update filename to reflect SVD use
                gmm_assignments_file = GMM_ASSIGNMENTS_FILE_TPL.format("TF-IDF_SVD")
                gmm_assignments_df.to_csv(gmm_assignments_file, index=False)
                print(f"    - GMM Assignments saved to: {gmm_assignments_file}")
                gmm_results['assignments_file'] = gmm_assignments_file
            except Exception as e:
                print(f"    Error saving GMM assignments: {e}")
                gmm_results['assignments_file'] = "Error"

        except MemoryError:
            print("  Error: MemoryError during GMM fitting/evaluation for TF-IDF. Skipping.")
            gmm_results = {'metrics': "Error (Memory)", 'assignments_file': "Skipped"}
        except Exception as e:
            print(f"  Error during GMM processing for TF-IDF: {e}")
            gmm_results = {'metrics': f"Error ({type(e).__name__})", 'assignments_file': "Skipped"}
    else:
        print("  TF-IDF representation not processed, skipping GMM.")

    # --- Section 5: Final Summary ---
    print(f"\n[{'4' if 'TF-IDF' in vectorizers_to_run else '3'}] Clustering Experiment Summary:") # Adjust section number

    for name, results in all_results.items():
        print(f"\n--- Results for K-Means: {name} ---")
        if 'metrics' in results and isinstance(results['metrics'], dict):
            metrics = results['metrics']
            print(f"  Metrics (Seed={SEED}):")
            print(f"    - Silhouette: {metrics.get('silhouette', 'N/A'):.4f}")
            print(f"    - Davies bouldin: {metrics.get('davies_bouldin', 'N/A'):.4f}")
            print(f"    - Ari: {metrics.get('ari', 'N/A'):.4f}")
            print(f"    - Homogeneity: {metrics.get('homogeneity', 'N/A'):.4f}")
            print(f"    - Completeness: {metrics.get('completeness', 'N/A'):.4f}")
            print(f"    - V measure: {metrics.get('v_measure', 'N/A'):.4f}")
            print(f"  Assignments File: {results.get('assignments_file', 'N/A')}")
            if 'tsne_plots' in results:
                 if isinstance(results['tsne_plots'], tuple):
                      print(f"  t-SNE Plots: {results['tsne_plots'][0]} (predicted), {results['tsne_plots'][1]} (true)")
                 else:
                      print(f"  t-SNE Plots: {results['tsne_plots']}") # Print error/skipped message
        else:
            print("  Metrics: Not available due to errors.")

    # Add GMM results to summary if available
    if gmm_results:
        print(f"\n--- Results for GMM: TF-IDF (with SVD n={SVD_N_COMPONENTS}) ---") # Updated title
        if 'metrics' in gmm_results and isinstance(gmm_results['metrics'], dict):
            metrics = gmm_results['metrics']
            print(f"  Metrics (Seed={SEED}):")
            print(f"    - Silhouette (on SVD data): {metrics.get('silhouette', 'N/A'):.4f}") # Clarify metric context
            print(f"    - Davies bouldin (on SVD data): {metrics.get('davies_bouldin', 'N/A'):.4f}") # Clarify metric context
            print(f"    - Ari: {metrics.get('ari', 'N/A'):.4f}")
            print(f"    - Homogeneity: {metrics.get('homogeneity', 'N/A'):.4f}")
            print(f"    - Completeness: {metrics.get('completeness', 'N/A'):.4f}")
            print(f"    - V measure: {metrics.get('v_measure', 'N/A'):.4f}")
            print(f"  Assignments File: {gmm_results.get('assignments_file', 'N/A')}")
        else:
            print(f"  Metrics: {gmm_results.get('metrics', 'Not available due to errors.')}")
            print(f"  Assignments File: {gmm_results.get('assignments_file', 'N/A')}")

    print(f"\n--- Clustering Experiment Finished ---") 