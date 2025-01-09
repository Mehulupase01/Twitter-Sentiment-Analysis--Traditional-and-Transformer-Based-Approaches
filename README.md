# Twitter Sentiment Analysis: Traditional and Transformer-Based Approaches
 This project compares machine learning (SVM, Naive Bayes), deep learning (LSTM), and transformer-based models (BERT, ROBERTA) for Twitter sentiment analysis. It pre-processes tweets, evaluates sentiment classification accuracy, and highlights transformer models' superior performance in nuanced text understanding

# Twitter Sentiment Analysis: Traditional and Transformer-Based Approaches

## Overview
This project explores sentiment analysis on Twitter data using various models:
1. **Traditional Machine Learning**: SVM, Naive Bayes.
2. **Deep Learning**: LSTM.
3. **Transformer Models**: BERT, ROBERTA.

The goal is to classify tweets as positive, negative, or neutral and compare these methodologies.

---

## Problem Statement
Social media platforms like Twitter are hubs for public expression. Sentiment analysis on Twitter involves classifying tweets into predefined sentiments. Challenges include:
- Sarcasm and idioms.
- Informal language and misspellings.

This project evaluates if transformer-based models significantly outperform traditional approaches.

---

## Goals

### Main Goals
- Compare performance across traditional, deep learning, and transformer models.
- Highlight the efficiency and accuracy trade-offs between methodologies.

### Sub-Goals
- Preprocess Twitter data for sentiment analysis.
- Implement advanced transformer-based models.
- Provide detailed evaluations for model comparison.

---

## Implementation

### Dataset
The dataset contains tweets with labeled sentiments (positive, negative, neutral), reflecting real-world informal language.

### Steps
1. **Data Preprocessing**:
   - Remove special characters, hashtags, and URLs.
   - Tokenize and normalize text.
2. **Model Implementation**:
   - Train models: SVM, Naive Bayes, LSTM, BERT, ROBERTA.
   - Evaluate on accuracy, precision, recall, and F1-score.
3. **Evaluation**:
   - Compare traditional vs. transformer approaches.
   - Analyze computational efficiency and accuracy trade-offs.

---

## Results

### Model Performance
| Model         | Accuracy | Precision | Recall | F1-Score |
|---------------|----------|-----------|--------|----------|
| SVM           | 84.47%   | 70.1%     | 68.9%  | 69.5%    |
| Naive Bayes   | 71.25%   | 67.0%     | 65.8%  | 66.4%    |
| LSTM          | 84.75%   | 78.5%     | 77.8%  | 78.1%    |
| BERT          | 98.31%   | 83.9%     | 84.1%  | 84.0%    |
| ROBERTA       | 95.19%   | 85.0%     | 85.2%  | 85.1%    |

---

## Inferences
1. **Traditional Models**: Provide a good baseline but struggle with nuanced language.
2. **Deep Learning**: LSTM improves contextual understanding but lacks transformer models' sophistication.
3. **Transformers**: BERT and ROBERTA excel in accuracy and F1-scores, effectively capturing language nuances.

### Key Insights
- Transformer models significantly outperform traditional and deep learning models.
- Computational efficiency is a trade-off for higher accuracy with transformer models.

---

## Files in Repository
- `preprocessing.py`: Script for cleaning and tokenizing tweets.
- `models.py`: Implementation of SVM, Naive Bayes, LSTM, BERT, and ROBERTA.
- `evaluation.py`: Script for evaluating models.
- `README.md`: Project documentation.

---

## References
- [BERT Documentation](https://huggingface.co/docs/transformers/model_doc/bert)
- [ROBERTA Documentation](https://huggingface.co/docs/transformers/model_doc/roberta)
- [Twitter Sentiment Dataset](https://www.kaggle.com/c/twitter-sentiment-analysis2/)

---

Feel free to contribute or raise issues!
