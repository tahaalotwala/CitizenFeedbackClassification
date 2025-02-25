import csv
import re
import math
import random
from collections import defaultdict, Counter


# Load dataset
def load_dataset(filepath):
    data = []
    with open(filepath, newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append((row['Comment'].lower(), row['Label']))
    return data


# Preprocess text
def tokenize(text):
    return re.findall(r'\b\w+\b', text)


# Split dataset into training and testing
def split_data(data, test_ratio=0.99):
    random.shuffle(data)
    split_index = int(len(data) * (1 - test_ratio))
    return data[:split_index], data[split_index:]


# Train Naïve Bayes classifier
def train_naive_bayes(train_data):
    label_counts = Counter(label for _, label in train_data)
    word_counts = {label: defaultdict(int) for label in label_counts}
    total_words = {label: 0 for label in label_counts}
    vocabulary = set()

    for text, label in train_data:
        words = tokenize(text)
        for word in words:
            word_counts[label][word] += 1
            total_words[label] += 1
            vocabulary.add(word)

    return label_counts, word_counts, total_words, len(vocabulary)


# Predict using Naïve Bayes
def predict(text, label_counts, word_counts, total_words, vocab_size):
    words = tokenize(text)
    label_probs = {}
    total_samples = sum(label_counts.values())

    for label in label_counts:
        log_prob = math.log(label_counts[label] / total_samples)
        for word in words:
            word_prob = (word_counts[label][word] + 1) / (
                    total_words[label] + vocab_size)  # Laplace smoothing
            log_prob += math.log(word_prob)
        label_probs[label] = log_prob

    return max(label_probs, key=label_probs.get)


# Evaluate model
def evaluate(test_data, label_counts, word_counts, total_words, vocab_size):
    correct = 0
    total = len(test_data)
    class_correct = Counter()
    class_total = Counter()

    for text, actual_label in test_data:
        predicted_label = predict(text, label_counts, word_counts, total_words,
                                  vocab_size)
        if predicted_label == actual_label:
            correct += 1
            class_correct[actual_label] += 1
        class_total[actual_label] += 1

    accuracy = correct / total
    print(f'Overall Accuracy: {accuracy:.2f}')
    for label in class_total:
        class_acc = class_correct[label] / class_total[label]
        print(f'Accuracy for {label}: {class_acc:.2f}')


# Main execution
if __name__ == "__main__":
    dataset_path = "datav2.csv"  # Ensure this file is present in the same directory
    dataset = load_dataset(dataset_path)
    train_set, test_set = split_data(dataset)

    label_counts, word_counts, total_words, vocab_size = train_naive_bayes(train_set)
    evaluate(test_set, label_counts, word_counts, total_words, vocab_size)
