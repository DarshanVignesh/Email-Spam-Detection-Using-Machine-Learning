# Email-Spam-Detection-Using-Machine-Learning

## Overview
This project aims to classify emails as spam or non-spam (ham) using machine learning techniques. It involves preprocessing email data, engineering features, training a classification model, and evaluating its performance.

## Dataset
The dataset used in this project consists of 5,728 emails obtained from various sources. It includes both spam and non-spam (ham) emails. The dataset has been preprocessed to remove duplicates and handle missing values.

## Preprocessing
- Duplicates Removal: Removed duplicate emails to ensure the integrity of the dataset.
- Missing Values Handling: Handled missing values to maintain data quality.
- Text Preprocessing: Removed punctuation and stopwords using the Natural Language Toolkit (NLTK) library.

## Feature Engineering
The text data was transformed into a matrix of token counts using the CountVectorizer from scikit-learn. This process converts the text into numerical features suitable for machine learning algorithms.

## Model Training
A Multinomial Naive Bayes classifier was trained on the preprocessed and feature-engineered dataset. The model was trained to learn patterns and relationships between features and target labels (spam or non-spam).

## Model Evaluation
The performance of the trained model was evaluated using various metrics such as accuracy, precision, recall, and F1-score. Additionally, confusion matrices were generated to visualize the model's performance on both the training and testing datasets.

## Results
- Accuracy on Training Dataset: 99.2%
- Accuracy on Testing Dataset: 98.8%

## Technologies Used
- Python
- NumPy
- pandas
- NLTK
- scikit-learn


