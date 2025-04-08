# TODO List

## Project Setup [x]
- [x] Initialize Git repository
- [x] Create project directory structure
- [x] Set up Python virtual environment
- [x] Set up .gitignore
- [x] Set up requirements.txt
- [x] Install required libraries
- [x] Create subdirectories
- [x] Create initial documentation files

## Data Loading and Exploration [x]
- [x] Obtain reduced dataset
- [x] Create data loading script
- [x] Perform initial EDA
- [x] Document dataset characteristics
- [x] Decide on text fields to use
- [x] Document rationale for text field selection

## Preprocessing [x]
- [x] Implement text cleaning functions
- [x] Create tokenization pipeline
- [x] Implement stopword removal
- [x] Implement stemming/lemmatization
- [x] Create vectorization functions:
  - [x] Binary
  - [x] Frequency
  - [x] TF-IDF
- [x] Build scikit-learn Pipeline
- [x] Test preprocessing pipeline

## Clustering [ ]
### k-means
- [x] Implement k-means with different random states
- [x] Create visualization functions
- [x] Implement internal evaluation metrics
- [x] Implement external evaluation metrics
- [x] Document results for each representation:
  - [x] Binary
  - [x] Frequency
  - [x] TF-IDF

### Gaussian Mixture
- [x] Implement Gaussian Mixture on TF-IDF
- [x] Run SVD n_components tuning script (`04_gmm_svd_param_tuning.py`)
- [x] Update main script (`03_...`) with best SVD n_components (No change needed, 100 confirmed)
- [x] Add TruncatedSVD step for TF-IDF before GMM (in main script)
- [x] Evaluate using same metrics as k-means
- [x] Compare with k-means results
- [x] Document findings

## Classification [ ]
### k-NN
- [x] Implement 5-fold Stratified Cross-Validation (in `05_...`)
- [x] Create parameter grid for k-NN (in `05_...`)
- [x] Test different combinations (in `05_...`):
  - [x] Binary representation
  - [x] Frequency representation
  - [x] TF-IDF representation
- [x] Save best k-NN models (joblib) (in `05_...`)
- [x] Document results (from `05_...`)

### Naïve Bayes
- [x] Create Naive Bayes script (`06_naive_bayes_classification.py`)
- [x] Implement MultinomialNB (in `06_...`)
- [x] Implement GaussianNB (in `06_...`, incl. dense handling)
- [x] Test both variants on all representations (in `06_...`)
- [x] Document results (from `06_...`)

### Comparison
- [ ] Combine results from `05_...` and `06_...`
- [ ] Create summary tables
- [ ] Analyze performance differences
- [ ] Document trade-offs

## Documentation [ ]
- [ ] Create report structure
- [ ] Write methodology section
- [ ] Document results and analysis
- [ ] Create visualizations
- [ ] Write conclusions
- [ ] Review and edit report

## Code Organization [ ]
- [ ] Clean up code
- [ ] Add comments
- [ ] Ensure reproducibility
- [ ] Create README
- [ ] Package for submission

## Final Steps [ ]
- [ ] Review all documentation
- [ ] Test code reproducibility
- [ ] Create submission package
- [ ] Submit before deadline 