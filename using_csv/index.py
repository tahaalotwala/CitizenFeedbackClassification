import csv
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# Function to preprocess text by removing punctuation and converting to lowercase
def preprocess_text(text):
    text = text.lower()
    text = ''.join(char if char.isalnum() or char.isspace() else ' ' for char in text)
    return text


# Load the data from data.csv
with open('datav2.csv', 'r') as file:
    reader = csv.DictReader(file)
    data = [row for row in reader]

# Initialize lists to store the feedback and categories
feedbacks = []
categories = []

# Populate the feedbacks and categories lists with data from data.csv
for entry in data:
    categories.append(entry['Label'])
    feedbacks.append(entry['Comment'])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(feedbacks, categories, test_size=0.99,
                                                    random_state=42)

# Initialize dictionaries for word frequencies by category
category_word_counts = defaultdict(lambda: defaultdict(int))


# Function to update word counts for training data
def update_word_counts(feedbacks, categories):
    for feedback, category in zip(feedbacks, categories):
        # Preprocess the feedback
        processed_feedback = preprocess_text(feedback)

        # Tokenize the feedback by splitting on spaces
        words = processed_feedback.split()

        # Count word frequencies in each category
        for word in words:
            category_word_counts[category][word] += 1


# Update the word counts using training data
update_word_counts(X_train, y_train)


# Function to classify new feedback
def classify_feedback(feedback):
    # Preprocess and tokenize the feedback
    processed_feedback = preprocess_text(feedback)
    words = processed_feedback.split()

    # Initialize category scores
    category_scores = defaultdict(int)

    # Calculate the score for each category based on word frequencies
    for word in words:
        for category, word_counts in category_word_counts.items():
            category_scores[category] += word_counts.get(word, 0)

    # Find the category with the highest score
    predicted_category = max(category_scores, key=category_scores.get)
    return predicted_category


# Classify the feedback from X_test and store predictions
y_pred = [classify_feedback(feedback) for feedback in X_test]

# Print the classification report
print(classification_report(y_test, y_pred))

# for i in range(len(y_test)):
#     print(y_test[i], y_pred[i])
