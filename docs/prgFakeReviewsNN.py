import numpy as np
import pandas as pd
import codecs
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

# Enable eager execution (Fixes RuntimeError)
tf.compat.v1.enable_eager_execution()

# Load dataset
data = pd.read_csv("reviews.csv")

# Drop unnecessary column
if "Unnamed: 0" in data.columns:
    data = data.drop(["Unnamed: 0"], axis=1)

# Encode labels (0 = Fake, 1 = Real)
le = LabelEncoder()
data['label'] = le.fit_transform(data['label'])

# Text preprocessing
titles = data['title'].astype(str).tolist()
labels = data['label'].tolist()

# Tokenization
tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(titles)
word_index = tokenizer.word_index
vocab_size = len(word_index) + 1  # +1 for padding

# Convert text to sequences and pad them
max_length = 54  # Max length of title sequence
padded_sequences = pad_sequences(tokenizer.texts_to_sequences(titles), maxlen=max_length, padding='post')

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.1, random_state=42)

# Convert lists to NumPy arrays
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

# Load pre-trained GloVe embeddings
embedding_dim = 50
embeddings_index = {}

with codecs.open('glove.6B.50d.txt', 'r', encoding='utf-8', errors='ignore') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

# Create embedding matrix
embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# Build the Model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim,
                              input_length=max_length, weights=[embedding_matrix], trainable=False),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv1D(64, 5, activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=4),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the Model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the Model
num_epochs = 10  # Adjust based on performance
history = model.fit(X_train, y_train, epochs=num_epochs, validation_data=(X_test, y_test), verbose=2)

# **SAVE the trained model**
model.save("fake_review_model.h5")

# **SAVE the tokenizer**
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("✅ Model and Tokenizer saved successfully!")
