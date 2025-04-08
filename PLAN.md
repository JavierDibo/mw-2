# Project Plan: Web Content Mining Practice

## 1. Project Setup
- Create project directory structure
- Set up Python virtual environment
- Install required libraries:
  - Core: scikit-learn, pandas, numpy
  - Visualization: matplotlib, seaborn
  - NLP: nltk (for preprocessing)
- Create subdirectories:
  - `/data`: For datasets
  - `/src`: For source code
  - `/notebooks`: For exploratory analysis
  - `/results`: For output files
  - `/report`: For documentation
- Initialize Git repository
- Create this PLAN.md and TODO.md

## 2. Data Loading and Initial Exploration
- Load reduced dataset (10k documents, 4 categories)
- Identify the 4 categories present
- Perform Exploratory Data Analysis (EDA):
  - Data shapes and basic statistics
  - Missing values analysis
  - Class distribution
  - Text length analysis
- Decide which text field(s) to use for analysis
  - Options: headline, text, shortdescription, or combinations
  - Document rationale for choice

## 3. Preprocessing Strategy & Pipeline
- Define preprocessing steps:
  - Text cleaning (remove HTML, special characters)
  - Tokenization
  - Lowercasing
  - Stopword removal (using standard English list)
  - Stemming (PorterStemmer) or Lemmatization
- Implement vectorization methods:
  - Binary (CountVectorizer with binary=True)
  - Frequency (CountVectorizer)
  - TF-IDF (TfidfVectorizer)
- Build scikit-learn Pipeline:
  - Combine preprocessing and vectorization
  - Set fixed random_state for reproducibility

## 4. Clustering Experiments
- Apply preprocessing pipeline for each representation:
  - Binary
  - Frequency
  - TF-IDF

### 4.1 k-means (K=4)
- For each representation:
  - Run multiple times with different random states
  - Analyze sensitivity to initialization
  - Select best run and save cluster assignments
  - Apply t-SNE for visualization
  - Calculate internal metrics:
    - Silhouette Score
    - Davies-Bouldin Score
  - Calculate external metrics:
    - Adjusted Rand Index
    - Homogeneity
    - Completeness
    - V-measure
  - Document and analyze results

### 4.2 Gaussian Mixture (n_components=4)
- Apply to TF-IDF representation only
- Evaluate using same metrics as k-means
- Compare with k-means results
- Document differences in cluster characteristics

## 5. Classification Experiments
- Implement 5-fold Stratified Cross-Validation
- Use fixed random_state for reproducibility

### 5.1 k-NN Classification
- For each representation:
  - Test combinations of:
    - n_neighbors: [3, 5, 7, 9]
    - weights: ['uniform', 'distance']
    - p: [1, 2]
  - Record mean cross-validated accuracy
  - Identify best parameters
  - Document results

### 5.2 Naïve Bayes Classification
- For each representation:
  - Test both MultinomialNB and GaussianNB
  - Record mean cross-validated accuracy
  - Compare performance between variants
  - Document results

### 5.3 Overall Comparison
- Create summary table of best results
- Compare algorithms and representations
- Discuss trade-offs:
  - Accuracy
  - Computational efficiency
  - Memory usage

## 6. Documentation and Reporting
- Structure PDF report:
  - Introduction
  - Methodology
  - Results and Analysis
  - Conclusions
- Include:
  - t-SNE visualizations
  - Performance tables
  - Code snippets
  - Justifications for choices
- Ensure code is:
  - Well-commented
  - Reproducible
  - Organized

## 7. Final Review and Submission
- Review report and code
- Package deliverables:
  - PDF report
  - Source code
- Submit via PLATEA before deadline (April 22nd, 23:59) 