# SpamEmailDetector

### This app is a spam detection app that can check whether a specific sentence is spam or not.
### It needs sample data that distinguishes whether it is spam or not.

# Features

### Uses the CountVectorizer from Scikit-learn to transform text data into numerical features for machine learning models
### Implements a Multinomial Naive Bayes classifier for email spam classification
### Utilizes the NLTK library for text preprocessing, including tokenization, stemming, and removal of stop words
### Provides a simple command-line interface for users to enter email text and receive a prediction of whether it is spam or not
### Computes and displays performance metrics such as accuracy, confusion matrix, and classification report for evaluating the model's performance
### Trains the model on a pre-labeled dataset of spam and non-spam emails for supervised learning
### Allows users to customize the maximum number of features and the test set size for the train-test split
