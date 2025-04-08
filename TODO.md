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

## Preprocessing [ ]
- [ ] Implement text cleaning functions
- [ ] Create tokenization pipeline
- [ ] Implement stopword removal
- [ ] Implement stemming/lemmatization
- [ ] Create vectorization functions:
  - [ ] Binary
  - [ ] Frequency
  - [ ] TF-IDF
- [ ] Build scikit-learn Pipeline
- [ ] Test preprocessing pipeline

## Clustering [ ]
### k-means
- [ ] Implement k-means with different random states
- [ ] Create visualization functions
- [ ] Implement internal evaluation metrics
- [ ] Implement external evaluation metrics
- [ ] Document results for each representation:
  - [ ] Binary
  - [ ] Frequency
  - [ ] TF-IDF

### Gaussian Mixture
- [ ] Implement Gaussian Mixture on TF-IDF
- [ ] Evaluate using same metrics as k-means
- [ ] Compare with k-means results
- [ ] Document findings

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