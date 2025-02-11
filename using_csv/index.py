import csv
from collections import defaultdict

# Function to preprocess text by removing punctuation and converting to lowercase
def preprocess_text(text):
    # Remove punctuation and convert to lowercase
    text = text.lower()
    text = ''.join(char if char.isalnum() or char.isspace() else ' ' for char in text)
    return text

# Load the data from data.csv
with open('data.csv', 'r') as file:
    reader = csv.DictReader(file)
    data = [row for row in reader]

# Initialize dictionaries for word frequencies by category
category_word_counts = defaultdict(lambda: defaultdict(int))

# Populate the category word counts with data from data.csv
for entry in data:
    category = entry['Category']
    feedback = entry['Feedback_Text']
    
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

# Example usage: Classify feedback from input.txt
with open('input.txt', 'r') as file:
    new_feedback = file.read().replace("\n", "")

# Classify the new feedback
predicted_category = classify_feedback(new_feedback)
print(f'The predicted category for the feedback is: {predicted_category}')
