import json
from collections import defaultdict

# Function to preprocess text by removing punctuation and converting to lowercase
def preprocess_text(text):
    # Remove punctuation and convert to lowercase
    text = text.lower()
    text = ''.join(char if char.isalnum() or char.isspace() else ' ' for char in text)
    return text

# Load the data from data.json (training data)
with open('train.json', 'r') as file:
    data = json.load(file)

# Load the test data from testjson (testing data)
with open('test.json', 'r') as file:
    test_data = json.load(file)

# Initialize dictionaries for word frequencies by category
category_word_counts = defaultdict(lambda: defaultdict(int))

# Populate the category word counts with data from data.json
for entry in data:
    category = entry['category']
    feedback = entry['feedback']
    
    # Preprocess the feedback
    processed_feedback = preprocess_text(feedback)
    
    # Tokenize the feedback by splitting on spaces
    words = processed_feedback.split()
    
    # Count word frequencies in each category
    for word in words:
        category_word_counts[category][word] += 1

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

# Function to calculate accuracy
def calculate_accuracy(test_data):
    correct_predictions = 0
    total_predictions = len(test_data)
    
    for entry in test_data:
        feedback = entry['feedback']
        actual_category = entry['category']
        
        # Get the predicted category
        predicted_category = classify_feedback(feedback)
        
        print(predicted_category, actual_category);
        
        # Compare predicted and actual category
        if predicted_category == actual_category:
            correct_predictions += 1
    
    # Calculate and return accuracy
    accuracy = correct_predictions / total_predictions * 100
    return accuracy

# Calculate accuracy on the test data
accuracy = calculate_accuracy(test_data)
print(f'The accuracy of the classifier is: {accuracy:.2f}%')
