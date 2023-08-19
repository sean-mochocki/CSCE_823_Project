import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow_hub import load
import graphviz
import pydot
import random
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Layer
import tensorflow.keras.losses
import tensorflow_similarity as tfsim
from keras.utils import plot_model
from tensorflow.keras.preprocessing.text import Tokenizer

Data_location = "/remote_home/AML/CSCE_823_Project/Data"
testFileName = "test_df.xlsx"
trainingFileName = "training_df.xlsx"
validationFileName = "validation_df.xlsx"

model_path = "AML/CSCE_823_Project/Models"

def read_excel_file(file_path, file_name, sheet_name=0):
    # Combine the file path and file name
    excel_file_path = f"{file_path}/{file_name}"
    
    # Read the Excel file and create a DataFrame
    df = pd.read_excel(excel_file_path, sheet_name=sheet_name)
    
    return df

test_df = read_excel_file(Data_location, testFileName)
training_df = read_excel_file(Data_location, trainingFileName)
validation_df = read_excel_file(Data_location, validationFileName)

# Function to convert a list of words to a sentence
def words_to_sentence(words_list):
    return ' '.join(words_list)

# Convert lists of words to full sentences
training_df['full_description'] = training_df['description'].apply(words_to_sentence)

# # Load the Universal Sentence Encoder
universal_sentence_encoder = load('https://tfhub.dev/google/universal-sentence-encoder/4')

# Function to encode sentences using the Universal Sentence Encoder
def encode_sentences(sentence):
    embeddings = universal_sentence_encoder([sentence])
    return embeddings

# Apply the encoding function to the 'full_description' column
training_df['embeddings'] = training_df['full_description'].apply(encode_sentences)


#break training_df into numpy arrays
# Convert 'cyber' column to numpy array
labels = training_df['cyber'].to_numpy()

# Convert 'embeddings' column to numpy array
embeddings = np.stack(training_df['embeddings'])

# Split data into positive and negative indices
positive_indices = np.where(labels == 1)[0]
negative_indices = np.where(labels == 0)[0]

# def triplet_semihard_loss(anchor, positive, negative, margin=1.0):
#     # Ensure anchor and positive have the same shape
#     anchor = tf.expand_dims(anchor, axis=1)
#     positive = tf.expand_dims(positive, axis=1)
    
#     distances = tf.norm(anchor - positive, axis=2)
#     negative_distances = tf.norm(anchor - negative, axis=2)
#     loss = tf.maximum(distances - negative_distances + margin, 0.0)
#     return loss

# Try the code with squared distances
def triplet_semihard_loss(anchor, positive, negative, margin=1.0):
    # Ensure anchor and positive have the same shape
    anchor = tf.expand_dims(anchor, axis=1)
    positive = tf.expand_dims(positive, axis=1)
    
    squared_distances = tf.reduce_sum(tf.square(anchor - positive), axis=2)
    squared_negative_distances = tf.reduce_sum(tf.square(anchor - negative), axis=2)
    loss = tf.maximum(squared_distances - squared_negative_distances + margin, 0.0)
    return loss

def triplet_loss_wrapper(y_true, y_pred):
    # Assuming y_true is not used in this loss function
    return tf.reduce_mean(triplet_semihard_loss(y_pred[0], y_pred[1], y_pred[2]))

# Create the basic triplet loss model 
# Create the anchor input
anchor_input = Input(shape=(512,))
positive_input = Input(shape=(512,))
negative_input = Input(shape=(512,))

# Create the embedding layer
embedding_layer = Dense(512)

# Create hidden layer
#hidden_layer = Dense(128, activation = 'relu')

# Encode the anchor, positive, and negative inputs
anchor_embedding = embedding_layer(anchor_input)
#anchor_embedding = hidden_layer(anchor_embedding)
positive_embedding = embedding_layer(positive_input)
#positive_embedding = hidden_layer(positive_embedding)
negative_embedding = embedding_layer(negative_input)
#negative_embedding = hidden_layer(negative_embedding)

# Calculate the triplet loss
loss = triplet_loss_wrapper(None, [anchor_embedding, positive_embedding, negative_embedding])

#print(model.summary())

# Calculate the triplet loss
model_output = {'loss': triplet_semihard_loss(anchor_embedding, positive_embedding, negative_embedding)}

# Create the model
model = Model(inputs=[anchor_input, positive_input, negative_input], outputs=model_output)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss=triplet_loss_wrapper)

# Define batch size and number of epochs
batch_size = 32
num_epochs = 100

# Create data generators for training
def generate_triplets(anchor_indices, positive_indices, negative_indices, batch_size):
    while True:
        batch_anchor = np.random.choice(anchor_indices, batch_size)
        batch_positive = np.random.choice(positive_indices, batch_size)
        batch_negative = np.random.choice(negative_indices, batch_size)
        
        yield [embeddings[batch_anchor], embeddings[batch_positive], embeddings[batch_negative]], np.zeros((batch_size,))

# Train the model
train_generator = generate_triplets(positive_indices, positive_indices, negative_indices, batch_size)

model.fit(train_generator, steps_per_epoch=len(positive_indices) // batch_size, epochs=num_epochs)