# Citizen Feedback Classification

## Overview

This project aims to classify citizen feedback into various categories such as **Compliment**, **Suggestion**, **Inquiry**, and **Complaint**. The goal is to automatically analyze and categorize feedback in order to streamline response efforts and identify areas for improvement or celebration in civic services.

## Problem Statement

Citizen feedback is essential for improving services, yet manually sorting this feedback can be time-consuming and inefficient. By automating the classification process, this project helps government agencies and organizations more effectively manage and respond to feedback, allowing for quicker and more accurate action.

## Methods Explored

### 1. **Bag of Words (BoW)**

Initially, a Bag of Words model was considered for converting text feedback into feature vectors. This method captures the frequency of words but disregards grammar and word order, making it a simple yet effective approach to text classification.

### 2. **Naive Bayes**

The Naive Bayes classifier, based on probability theory, was tested as a potential solution. It is a popular text classification algorithm that assumes features (words) are independent and applies Bayes' Theorem to calculate the probability of each feedback belonging to a particular class.

### 3. **Multiclass Logistic Regression with Sigmoid Activation**

After evaluating various methods, the project settled on a **Multiclass Logistic Regression** approach, employing the **Sigmoid activation function**. This model was chosen due to its simplicity, scalability, and ability to handle multiple classes efficiently. The logistic regression model uses a linear combination of features and applies the sigmoid function to predict the probability of each feedback belonging to one of the categories (Compliment, Suggestion, Inquiry, Complaint).

## Model Architecture

The model uses **Logistic Regression** with the following key steps:

- **Feature Extraction**: The feedback text is first preprocessed, with tokenization and vectorization (using TF-IDF ) to convert textual data into numerical form.
- **Training**: The processed data is then passed to a multiclass logistic regression model, which is trained to predict the class probabilities for each piece of feedback.
- **Prediction**: The sigmoid activation function is used to convert the output of the logistic regression model into probabilities, and the class with the highest probability is assigned as the predicted category.

## Performance Evaluation

The performance of the model was evaluated using standard metrics such as:

- **Accuracy**: The percentage of correctly classified feedback instances.
- **Precision, Recall, and F1-Score**: For each class, these metrics were computed to assess the model's effectiveness in handling each category.

The final model offers a balance between precision and recall, making it well-suited for real-world applications where all categories are important to address.

## Key Insights

- **Flexibility of Multiclass Logistic Regression**: Logistic Regression is a robust classifier that can handle multiclass problems efficiently with minimal complexity.
- **Improved Categorization**: The model was able to distinguish between nuanced feedback types, enabling better segmentation of citizen feedback.
- **Real-World Applicability**: The results suggest that automated classification can significantly reduce manual labor in processing citizen feedback, improving response times and ensuring more focused attention to each type of feedback.

## Potential Future Enhancements

- **Deep Learning**: Experimenting with neural networks, such as LSTM (Long Short-Term Memory) or Transformer models, could further enhance classification accuracy, especially in complex feedback scenarios.
- **Multilingual Support**: Expanding the model to handle feedback in multiple languages could help extend its usability.
- **Sentiment Analysis**: Combining the classification model with sentiment analysis could provide deeper insights into the nature of the feedback.

## Conclusion

This project successfully applies text classification techniques to automate the categorization of citizen feedback. By leveraging machine learning models like logistic regression with sigmoid activation, the system offers a reliable and scalable solution for processing large volumes of feedback, ultimately helping organizations better understand and respond to the needs of citizens.
