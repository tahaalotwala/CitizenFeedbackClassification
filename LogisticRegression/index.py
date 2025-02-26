import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score

# Load datasets
df_train = pd.read_csv("../data/datav3.csv")
df_test = pd.read_csv("../data/test.csv")

# Extract features and labels
X_train = df_train["Comment"]
y_train = df_train["Label"].map(
    {"Complaint": 0, "Inquiry": 1, "Suggestion": 2, "Compliment": 3})
X_test = df_test["Comment"]
y_test = df_test["Label"].map(
    {"Complaint": 0, "Inquiry": 1, "Suggestion": 2, "Compliment": 3})

# Convert text to TF-IDF features
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_tfidf = tfidf.fit_transform(X_train).toarray()
X_test_tfidf = tfidf.transform(X_test).toarray()


# Initialize weights and bias
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # Numerical stability
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def train_logistic_regression(X, y, lr=0.01, epochs=1000):
    m, n = X.shape
    k = len(np.unique(y))  # Number of classes
    weights = np.zeros((n, k))
    bias = np.zeros(k)
    y_one_hot = np.eye(k)[y]  # One-hot encoding

    for _ in range(epochs):
        linear_model = np.dot(X, weights) + bias
        predictions = softmax(linear_model)

        error = predictions - y_one_hot

        weights -= lr * np.dot(X.T, error) / m
        bias -= lr * np.mean(error, axis=0)

    return weights, bias


def predict(X, weights, bias):
    linear_model = np.dot(X, weights) + bias
    predictions = softmax(linear_model)
    return np.argmax(predictions, axis=1)


# Train model
weights, bias = train_logistic_regression(X_train_tfidf, y_train.to_numpy())

# Predictions
y_pred = predict(X_test_tfidf, weights, bias)

# Convert numerical predictions back to labels
label_map = {0: "Complaint", 1: "Inquiry", 2: "Suggestion", 3: "Compliment"}
y_pred_labels = np.array([label_map[val] for val in y_pred])
y_test_labels = np.array([label_map[val] for val in y_test])

# Print results
print("Accuracy on Training Data:",
      accuracy_score(y_train, predict(X_train_tfidf, weights, bias)))
print("Accuracy on Testing Data:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test_labels, y_pred_labels))
