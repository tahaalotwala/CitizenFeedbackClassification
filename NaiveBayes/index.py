import os
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report

from utils import stopwords

df = pd.read_csv("data.csv")
n = (len(df["Category"]))
x = []
y = []
for i in range(0, n):
    x.append(df["Feedback_Text"][i])
    y.append(df["Category"][i])

x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.25,
                                                                    random_state=0)

vocab = {}
for i in range(len(x_train)):
    word_list = []
    for word in x_train[i].split():
        new_word = word.strip(string.punctuation).lower()
        if (len(new_word) > 2) and (new_word not in stopwords):
            if new_word in vocab:
                vocab[new_word] += 1
            else:
                vocab[new_word] = 1

num_words = [0 for i in range(max(vocab.values()) + 1)]
freq = [i for i in range(max(vocab.values()) + 1)]
for key in vocab:
    num_words[vocab[key]] += 1
plt.plot(freq, num_words)
# plt.axis((1, 10, 0, 100))
plt.xlabel("Frequency")
plt.ylabel("No. of words")
plt.grid()
plt.show()
