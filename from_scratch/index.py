import string;

text = open('input.txt', encoding="utf-8").read();
cleaned_text = text.lower().replace("\n", "").translate(str.maketrans("", "", string.punctuation));

tokenized_text = cleaned_text.split();

stop_words = open("stop_words.txt", encoding="utf-8").read().split(", ");

tokens = [word for word in tokenized_text if word not in stop_words ];

properties = [];
test = {};

with open('categories.txt', 'r', encoding='utf-8') as file : 
  for line in file : 
    cleaned_line = line.replace("\n", "").replace(",", "").replace("'", "").strip();
    word, emotion = cleaned_line.split(" : ");
    
    if word in tokens : 
      test[word] = emotion;
      properties.append(emotion);
      
print(properties);
print(test);