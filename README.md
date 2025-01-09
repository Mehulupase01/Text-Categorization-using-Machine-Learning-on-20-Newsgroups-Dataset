# Text Categorization using Machine Learning on 20 Newsgroups Dataset
 This repository contains a comprehensive project on text categorization using machine learning, leveraging the 20 Newsgroups dataset. The task includes evaluating three classifiers (Multinomial Naive Bayes, Random Forest, Support Vector Machine) and three feature extraction methods (Count, TF, TF-IDF). The optimal combination of Random Forest with TF-IDF was identified for superior performance. Detailed experiments explore hyperparameters, feature effects, and classifier efficiency, with results presented in terms of Precision, Recall, and F1-score. Scripts, results, and documentation are included for reproducibility and further exploration.

# Text Categorization using Machine Learning on 20 Newsgroups Dataset

## Overview
This project explores text categorization using machine learning methods on the 20 Newsgroups dataset. The primary objective is to evaluate the performance of different classifiers and feature extraction techniques, culminating in identifying the best combination for accurate text classification.

---

## Problem Statement
The task involves classifying text documents into one of 20 categories in the 20 Newsgroups dataset. The project aims to:

1. Utilize three different classifiers: Multinomial Naive Bayes, Random Forest, and Support Vector Machine.
2. Experiment with three feature extraction methods: Count, Term Frequency (TF), and Term Frequency-Inverse Document Frequency (TF-IDF).
3. Evaluate performance based on Precision, Recall, and F1-score.

---

## Goals

### Main Goals
- Develop a robust text categorization pipeline using scikit-learn.
- Evaluate the effect of different feature extraction techniques on classifier performance.

### Sub-Goals
- Experiment with hyperparameters in the `CountVectorizer` function (e.g., `lowercase`, `stop_words`, `analyzer`, `max_features`).
- Identify the optimal combination of classifier and feature extraction method.

---

## Implementation

### Dataset
The 20 Newsgroups dataset is used, which consists of documents organized into 20 categories. Preprocessing included removal of headers, footers, and quotes.

### Classifiers
1. **Multinomial Naive Bayes (MNB)**
   - A probabilistic learning method leveraging Bayes' theorem.
   - Predicts tags by calculating probabilities of each tag and selecting the highest.
2. **Random Forest**
   - An ensemble method building multiple decision trees.
   - Combines predictions to achieve higher accuracy and stability.
3. **Support Vector Machine (SVM)**
   - Finds an optimal hyperplane for classification using kernel transformations.
   - Suitable for complex data and provides robust performance with TF-IDF.

### Feature Extraction
1. **Count**: Converts text into a matrix of token counts.
2. **Term Frequency (TF)**: Represents the frequency of terms in documents.
3. **TF-IDF**: Weighs terms based on importance in both the document and corpus.

---

## Experimental Setup
1. Compared the three classifiers (MNB, Random Forest, SVM) using all three feature extraction methods.
2. Tuned the `CountVectorizer` parameters for the best-performing classifier-feature combination:
   - `lowercase`: True/False
   - `stop_words`: With/Without
   - `analyzer`: Tested different analyzers with `ngram_range`
   - `max_features`: Experimented with various values

### Results Table
| Classifier                | Feature Extraction | Precision | Recall | F1-Score |
|---------------------------|--------------------|-----------|--------|----------|
| Multinomial Naive Bayes   | Count             | 0.87      | 0.85   | 0.84     |
| Multinomial Naive Bayes   | TF                | 0.83      | 0.77   | 0.75     |
| Multinomial Naive Bayes   | TF-IDF            | 0.88      | 0.85   | 0.84     |
| Random Forest             | Count             | 0.85      | 0.85   | 0.85     |
| Random Forest             | TF                | 0.85      | 0.84   | 0.84     |
| Random Forest             | TF-IDF            | 0.95      | 0.45   | 0.61     |
| Support Vector Machine    | Count             | 0.06      | 0.05   | 0.01     |
| Support Vector Machine    | TF                | 0.86      | 0.85   | 0.85     |
| Support Vector Machine    | TF-IDF            | 0.91      | 0.91   | 0.91     |

---

## Inferences
1. **Random Forest**
   - Showed consistent performance with all features.
   - Best results achieved with TF-IDF.
2. **Multinomial Naive Bayes**
   - Performed well but struggled with TF features.
3. **Support Vector Machine**
   - Outperformed others with TF-IDF but failed significantly with Count features.
4. **Feature Extraction**
   - TF-IDF consistently outperformed Count and TF due to its ability to capture term importance.

### Conclusion
The optimal combination for text classification is **Random Forest with TF-IDF**, demonstrating the highest Precision, Recall, and F1-scores. Future tasks with similar datasets are recommended to utilize **Support Vector Machine with TF-IDF** for better accuracy.

---

## How it Works

### Algorithms
1. **Multinomial Naive Bayes**
   - Based on Bayes' theorem.
   - Suitable for categorical data.
2. **Random Forest**
   - Combines multiple decision trees for robust predictions.
3. **Support Vector Machine**
   - Utilizes kernel tricks for high-dimensional space classification.

### Mathematical Equations
1. **TF-IDF Weighting**:
   \[
   \text{TF-IDF}(t, d) = \text{TF}(t, d) \times \log \left( \frac{N}{\text{DF}(t)} \right)
   \]
   Where:
   - \( t \): Term
   - \( d \): Document
   - \( N \): Total number of documents
   - \( \text{DF}(t) \): Number of documents containing term \( t \)

---

## Files in Repository
- `main.py`: Main script for running experiments
- `results.csv`: Stores performance metrics
- `README.md`: Project documentation

---

## References
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [20 Newsgroups Dataset](http://qwone.com/~jason/20Newsgroups/)

---

Feel free to contribute or raise issues!
