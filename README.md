#Movie Genre Prediction System

This project predicts the genre of a movie based on its plot summary using Natural Language Processing (NLP) and a Logistic Regression model.

## Overview
- Dataset: Movie metadata and plot summaries
- Model: One-vs-Rest Logistic Regression
- Features: TF-IDF vectors (10,000 most frequent words)
- Evaluation Metric: Micro F1-Score

##Steps Implemented
1. Load and preprocess datasets from Google Drive  
2. Clean and tokenize text using NLTK  
3. Remove stopwords  
4. Extract TF-IDF features  
5. Train a Logistic Regression model using One-vs-Rest strategy  
6. Evaluate model using F1-score  
7. Implement inference function to predict genres for new plots

## Example Code Snippet
```python
def infer_tags(q):
    q = clean_text(q)
    q = remove_stopwords(q)
    q_vec = tfidf_vectorizer.transform([q])
    q_pred = clf.predict(q_vec)
    return multilabel_binarizer.inverse_transform(q_pred)
