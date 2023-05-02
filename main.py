# Project Name: Spam Email Detector
# Description: This program detects if an email is spam (1) or not (0)
# Name: James Kim,  Email: jamesmall153@gmail.com
# Due Date: April 4, 2023


# Import the necessary libraries
# Pandas: pip install pandas
# Scikit-learn: pip install scikit-learn
# NLTK: pip install nltk
# Matplotlib: pip install matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the spamCollection.csv file into a pandas DataFrame
df = pd.read_csv("spamCollection.csv")
print(df.head())

# Convert the labels to numerical values (0 for ham, 1 for spam)
# pd.get_dummies(df["label"]): This function creates dummy variables for the label column of the DataFrame df. 
# It creates a new DataFrame where each unique value in df["label"] is converted to a new column with a binary value of 0 or 1. 
# In this case, there are two unique values in df["label"]: "ham" and "spam", 
# so the resulting DataFrame will have two columns: one for "ham" and one for "spam".
# pd.get_dummies(df["label"])["spam"]: This selects the "spam" column from the dummy variable DataFrame, 
# which contains binary values indicating whether each row in the original df["label"] column was "spam" or not.
df["label"] = pd.get_dummies(df["label"])["spam"]

# Split the dataset into training and testing sets
# X_train: This is the training data for the model, containing the email text.
# X_test: This is the testing data for the model, also containing the email text.
# y_train: This is the training labels for the model, containing binary values indicating whether each email is spam or ham (0 for ham, 1 for spam).
# y_test: This is the testing labels for the model, also containing binary values indicating whether each email is spam or ham.
# test_size=0.2, which specifies that 20% of the data should be used for testing and 80% for training.
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2)

# Define a CountVectorizer object to convert emails to a matrix of token counts
# stop_words="english": common English words (such as "the", "a", and "an") should be removed from the email text 
# before counting the occurrences of each remaining word. 
# This is because these words are not generally informative for distinguishing between spam and ham emails.
# max_features=5000: the vectorizer should only consider the 5000 most common words in the email text 
# when creating the numerical representations. This is done to limit the dimensionality of the feature space and avoid overfitting.
vectorizer = CountVectorizer(stop_words="english", max_features=5000)

# Fit the vectorizer on the training data and transform the training data
X_train_counts = vectorizer.fit_transform(X_train)

# Train the model on the training data
# creates a Naive Bayes classifier with a multinomial distribution.
# this classifier to train a model on the email data and then use the model to predict whether a new email is spam or not. 
clf = MultinomialNB()

# X_train_counts: count-based features of the training data, 
# y_train: corresponding labels.
clf.fit(X_train_counts, y_train)

# Apply the same vectorizer to transform the testing data
X_test_counts = vectorizer.transform(X_test)

# Predict the labels of the test data
y_pred = clf.predict(X_test_counts)

# Evaluate the model's performance
# accuracy will be a float between 0 and 1, 
# where 1 represents 100% accuracy (i.e., all test emails were classified correctly) and 
# 0 represents 0% accuracy (i.e., none of the test emails were classified correctly).
accuracy = accuracy_score(y_test, y_pred)

# computes the confusion matrix of the spam classifier model by comparing the predicted labels (y_pred) 
# with the actual labels (y_test) for the test set of emails. 
# 2x2 NumPy array where the rows represent the actual labels (ham and spam) and 
# the columns represent the predicted labels (ham and spam)
# TP = confusion_mat[1, 1]  # True positives (spam emails classified as spam)
# TN = confusion_mat[0, 0]  # True negatives (ham emails classified as ham)
# FP = confusion_mat[0, 1]  # False positives (ham emails classified as spam)
# FN = confusion_mat[1, 0]  # False negatives (spam emails classified as ham)
confusion_mat = confusion_matrix(y_test, y_pred)

# generates a text report of the main classification metrics for a given set of predicted and actual labels.
# Precision is the ratio of correctly predicted positive observations (true positives) 
# to the total predicted positives (true positives and false positives). 
# Recall is the ratio of correctly predicted positive observations (true positives) 
# to the total actual positives (true positives and false negatives). 
# F1-score is the weighted average of precision and recall, 
# where the best score is 1.0 and the worst is 0.0. 
# Support is the number of occurrences of each class in the actual labels.
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.3f}")
print('Confusion matrix:\n', confusion_mat)
print('Classification report:\n', report)

# Visualize the performance of the model
plt.bar(["Accuracy"], [accuracy], color="blue")

# sets the range of the y-axis to be between 0.9 and 1.0 
# to better visualize the small differences in accuracy values.
plt.ylim([0.9, 1.0])
plt.ylabel("Score")
plt.title("Model Performance")
plt.show()


# Prompt the user to enter email contents and preprocess them
for i in range(3):
    new_email = input(f"Enter email {i+1} content: ")

    # PorterStemmer: to "stem" each word. 
    # Stemming involves reducing words to their root form (e.g., "running", "runs", and "ran" are all stemmed to "run"). 
    # This is done to reduce the dimensionality of the feature space and avoid overfitting.
    new_email_processed = ' '.join(PorterStemmer().stem(word.lower()) for word in new_email.split() if word.lower() not in stopwords.words('english'))

    # Use the classifier to predict whether the new email is spam or not
    # It converts the processed new email string in new_email_processed into a matrix of word counts, 
    # where each row corresponds to the new email and each column corresponds to a unique word/token in the training data. 
    # The values in the matrix represent the number of times each word appears in the new email.
    new_email_counts = vectorizer.transform([new_email_processed])
    
    # trained on the training data using the fit method, 
    # which learns the relationship between the word counts in each email and the label (spam or not) associated with that email. After training, the classifier can be used to predict the labels of new emails using the predict method.
    clf = MultinomialNB()

    # It learns the relationship between the word counts in each email (represented as the columns of X_train_counts) and 
    # the label associated with that email (stored in y_train). Specifically, 
    # it computes the probability of each word occurring in spam emails and non-spam emails based on the training data, and 
    # uses these probabilities to make predictions on new, unseen data.
    clf.fit(X_train_counts, y_train)    

    # It takes as input a matrix of word counts for one or more new emails, and 
    # returns an array of predicted labels (spam or not) for each input email.
    prediction = clf.predict(new_email_counts)[0]

    # If the predicted value is True (i.e., the email is predicted to be spam), the label variable is set to "spam". 
    # If the predicted value is False (i.e., the email is predicted to be not spam), the label variable is set to "ham"
    label = "spam" if prediction else "ham"

    print(f"The email {i+1} is {label}\n")


"""
# examples

Not spam: "Hey, are you free for lunch tomorrow?"
Spam: "Congratulations! You've been selected for a free trial of our miracle weight loss pill!"
Not spam: "Just a friendly reminder that our team meeting is at 2:00 PM today."
Spam: "URGENT: Your account has been suspended. Click here to verify your information and reactivate your account."
"""