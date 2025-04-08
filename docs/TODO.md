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
- [x] Add TruncatedSVD step for TF-IDF before GMM
- [x] Evaluate using same metrics as k-means
- [x] Compare with k-means results
- [x] Document findings

## Classification [ ]
### k-NN
- [ ] Implement 5-fold Stratified Cross-Validation
- [ ] Create parameter grid for k-NN
- [ ] Test different combinations:
  - [ ] Binary representation
  - [ ] Frequency representation
  - [ ] TF-IDF representation
- [ ] Document results

### Naïve Bayes
- [ ] Implement MultinomialNB
- [ ] Implement GaussianNB
- [ ] Test both variants on all representations
- [ ] Document results

### Comparison
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