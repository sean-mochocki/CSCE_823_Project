import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow_hub import load
from keras.utils.vis_utils import plot_model
#from tensorflow.keras.utils import plot_model

import pydot
import random
from sklearn.metrics import confusion_matrix
import seaborn as sns
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Layer
import tensorflow.keras.losses
#import tensorflow_similarity as tfsim
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.metrics import confusion_matrix, f1_score, recall_score, balanced_accuracy_score
from tensorflow.keras.models import load_model

Data_location = "/remote_home/AML/CSCE_823_Project/Data"
testFileName = "test_df_with_title.xlsx"
trainingFileName = "training_df_with_title.xlsx"
validationFileName = "validation_df_with_title.xlsx"
anchorDataFileName = "anchor_data_updated_processed.xlsx"
twenty_newsgroupFileName = "twenty_newsgroup_df.xlsx"

model_path = "/remote_home/AML/CSCE_823_Project/Models"
model_name = "siamese_model.h5"
figure_path = "/remote_home/AML/CSCE_823_Project/Figures"

def plot_loss(validation_losses, plot_name):
    plt.figure(figsize=(10, 6))

    for i, val_loss_values in enumerate(validation_losses):
        plt.plot(val_loss_values, label=f'Session {i+1}')

    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.title(plot_name)
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{figure_path}/{plot_name}.png', dpi=300, bbox_inches='tight')

def plot_confusion_matrix(y_true, y_pred, save_location=None, figure_name=None, normalize=False, validation=True):
    conf_matrix = confusion_matrix(y_true, y_pred, normalize='true' if normalize else None)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='.2f' if normalize else 'd', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])

    if validation and figure_name:
        figure_name = figure_name + " validation"
    else:
        figure_name = figure_name + " test"

    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(figure_name)

    if save_location and figure_name:
        plt.savefig(f'{save_location}/{figure_name}.png', dpi=300, bbox_inches='tight')


def read_excel_file(file_path, file_name, sheet_name=0):
    # Combine the file path and file name
    excel_file_path = f"{file_path}/{file_name}"
    
    # Read the Excel file and create a DataFrame
    df = pd.read_excel(excel_file_path, sheet_name=sheet_name)
    
    return df



def process_description_to_embeddings(dataframe, columnName):
  # Process description into UTE
  # Function to convert a list of words to a sentence
  def words_to_sentence(words_list):
      return ' '.join(words_list)

  # Convert lists of words to full sentences
  dataframe['full_description'] = dataframe[columnName].apply(words_to_sentence)

  # # Load the Universal Sentence Encoder
  universal_sentence_encoder = load('https://tfhub.dev/google/universal-sentence-encoder/4')

  # Function to encode sentences using the Universal Sentence Encoder
  def encode_sentences(sentence):
      embeddings = universal_sentence_encoder([sentence])
      return embeddings

  # Apply the encoding function to the 'full_description' column
  dataframe['embeddings'] = dataframe['full_description'].apply(encode_sentences)

  return dataframe



def get_positive_negative_indices(column_name, dataframe):
    labels = dataframe[column_name].to_numpy()
    positive_indices = np.where(labels == 1)[0]
    negative_indices = np.where(labels == 0)[0]
    return positive_indices, negative_indices

# # Get the positive and negative indices for the training set
# column_name = 'cyber'
# positive_indices, negative_indices = get_positive_negative_indices(column_name, training_df)

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

def create_model(model, learning_rate = 0.001):
    """ 
    Pass in an empty model, and this function will return a populated model
    """
    # Create the basic triplet loss model 
    # Create the anchor input
    anchor_input = Input(shape=(512,), name = 'Anchor')
    positive_input = Input(shape=(512,), name = 'Positive Input')
    negative_input = Input(shape=(512,), name = 'Negative Input')

    # Hidden layer shared for all inputs
    hidden_layer = Dense(1000, activation='relu', name = 'Shared_Hidden_Layer')

    # Apply the hidden layer to each input
    anchor_hidden = hidden_layer(anchor_input)
    positive_hidden = hidden_layer(positive_input)
    negative_hidden = hidden_layer(negative_input)

    # Create the embedding layer
    embedding_layer = Dense(512, name = 'Embedding_Layer')

    # Encode the anchor, positive, and negative hidden inputs
    anchor_embedding = embedding_layer(anchor_hidden)
    positive_embedding = embedding_layer(positive_hidden)
    negative_embedding = embedding_layer(negative_hidden)

    # Wrap the triplet loss
    loss = triplet_loss_wrapper(None, [anchor_embedding, positive_embedding, negative_embedding])

    # Calculate the triplet loss
    model_output = {'loss': triplet_semihard_loss(anchor_embedding, positive_embedding, negative_embedding)}

    # Create the model
    model = Model(inputs=[anchor_input, positive_input, negative_input], outputs=model_output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss=triplet_loss_wrapper)

    return model

# Create data generators for training
def generate_triplets(anchor_indices, positive_indices, negative_indices, batch_size):
    while True:
        batch_anchor = np.random.choice(anchor_indices, batch_size)
        batch_positive = np.random.choice(positive_indices, batch_size)
        batch_negative = np.random.choice(negative_indices, batch_size)
          
        yield [training_embeddings[batch_anchor], training_embeddings[batch_positive], training_embeddings[batch_negative]], np.zeros((batch_size,))

def train_model(model, columnName, dataframe, batch_size = 32, epochs = 5):
  validation_losses = []
  for train_column_name in column_names:
    # Get the positive and negative indices for the training set
    positive_indices_train, negative_indices_train = get_positive_negative_indices(train_column_name, training_df)

    # Split your data into training and validation sets
    # Example: positive_indices_train, positive_indices_val = train_test_split(positive_indices, test_size=0.2)
    
    for val_column_name in column_names:
        if val_column_name == train_column_name:
            continue  # Skip using the same label for validation
        
        # Get the positive and negative indices for the validation set
        positive_indices_val, negative_indices_val = get_positive_negative_indices(val_column_name, training_df)

        # Create data generators for training and validation
        train_generator = generate_triplets(positive_indices_train, positive_indices_train, negative_indices_train, batch_size)
        val_generator = generate_triplets(positive_indices_val, positive_indices_val, negative_indices_val, batch_size)

        history = model.fit(
            train_generator,
            steps_per_epoch=len(positive_indices_train) // batch_size,
            epochs=epochs,
            validation_data=val_generator,
            validation_steps=len(positive_indices_val) // batch_size,
        )

        validation_losses.append(history.history['val_loss'])

    return model, validation_losses

#Declare an empty model
model = None

train_model_twentyNewsgroup = True

if train_model_twentyNewsgroup:
    #training_df_20 = pd.read_excel(Data_location, twenty_newsgroupFileName, nrows=500)

    file_path = '/remote_home/AML/CSCE_823_Project/Files/20Newsgroup_training_embeddings.npy'
    # Before processing description and embeddings
    try:
        training_embeddings = np.load(file_path)
        print("Embeddings loaded successfully.")
    except FileNotFoundError:
        print("Embeddings file not found. Recreating embeddings...")

    training_df_20 = read_excel_file(Data_location, twenty_newsgroupFileName)
    training_df_20 = process_description_to_embeddings(training_df_20, 'data')
    training_embeddings = np.stack(training_df_20['embeddings'])
    np.save(file_path, training_embeddings)



    model = create_model(model, learning_rate = 0.001)

    column_names = ['alt.atheism',	'comp.graphics',	'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware',
    	'comp.sys.mac.hardware',	'comp.windows.x',	'misc.forsale',	'rec.autos',	'rec.motorcycles',	
        'rec.sport.baseball',	'rec.sport.hockey',	'sci.crypt',	'sci.electronics',	'sci.med',	'sci.space',
        'soc.religion.christian', 'talk.politics.guns',	'talk.politics.mideast',	'talk.politics.misc',	'talk.religion.misc']

    batch_size = 32
    num_epochs = 100 

    for column_name in column_names:
        # Get the positive and negative indices for the training set
        positive_indices, negative_indices = get_positive_negative_indices(column_name, training_df_20)

        total_examples = len(positive_indices)
        steps_per_epoch = total_examples//batch_size
        if total_examples % batch_size != 0:
            steps_per_epoch += 1  # Account for the last smaller batch

        # Train the model
        train_generator = generate_triplets(positive_indices, positive_indices, negative_indices, batch_size)
        model.fit(train_generator, steps_per_epoch=steps_per_epoch, epochs=num_epochs)

    model.save(f'{model_path}/{model_name}')

train_model_Avolve = False

if train_model_Avolve:
  #Read dataframes from excel files
  training_df = read_excel_file(Data_location, trainingFileName)
  training_df = process_description_to_embeddings(training_df, 'description')
  training_embeddings = np.stack(training_df['embeddings'])

  model = create_model(model, learning_rate = 0.001)
  
  plot_model(model, to_file='./AML/CSCE_823_Project/Figures/mod.png', show_shapes=True, show_layer_names=True)

  column_names = ['cyber', 'innovation', 'acquisition', 'cybersecurity', 'toc', 
  'intelligence', 'python data science', 'digital engineering']



  validation_losses = []
  #model, validation_losses = train_model(model, column_names, training_df, batch_size=16, epochs = 20)

  batch_size = 8
  num_epochs = 20 

  for column_name in column_names:
    # Get the positive and negative indices for the training set
    positive_indices, negative_indices = get_positive_negative_indices(column_name, training_df)

    total_examples = len(positive_indices)
    steps_per_epoch = total_examples//batch_size
    if total_examples % batch_size != 0:
        steps_per_epoch += 1  # Account for the last smaller batch

    # Train the model
    train_generator = generate_triplets(positive_indices, positive_indices, negative_indices, batch_size)
    model.fit(train_generator, steps_per_epoch=steps_per_epoch, epochs=num_epochs)

#   for train_column_name in column_names:
#     # Get the positive and negative indices for the training set
#     positive_indices_train, negative_indices_train = get_positive_negative_indices(train_column_name, training_df)

#     # Split your data into training and validation sets
#     # Example: positive_indices_train, positive_indices_val = train_test_split(positive_indices, test_size=0.2)
    
#     for val_column_name in column_names:
#         if val_column_name == train_column_name:
#             continue  # Skip using the same label for validation
        
#         # Get the positive and negative indices for the validation set
#         positive_indices_val, negative_indices_val = get_positive_negative_indices(val_column_name, training_df)

#         # Create data generators for training and validation
#         train_generator = generate_triplets(positive_indices_train, positive_indices_train, negative_indices_train, batch_size)
#         val_generator = generate_triplets(positive_indices_val, positive_indices_val, negative_indices_val, batch_size)

#         history = model.fit(
#             train_generator,
#             steps_per_epoch=len(positive_indices_train) // batch_size,
#             epochs=num_epochs,
#             validation_data=val_generator,
#             validation_steps=len(positive_indices_val) // batch_size,
#         )

#         validation_losses.append(history.history['val_loss'])


  #plot_loss(validation_losses = validation_losses, plot_name = 'Training Losses')
  model.save(f'{model_path}/{model_name}')  

else:
  model = load_model(f'{model_path}/{model_name}', custom_objects={'triplet_loss_wrapper': triplet_loss_wrapper})


validation = True
if validation:
    #Next validate the model, pull in the model database
    validation_df = read_excel_file(Data_location, validationFileName)
    anchor_df = read_excel_file(Data_location, anchorDataFileName)

    #Get the validation embeddings
    validation_df = process_description_to_embeddings(validation_df, 'description')
    validation_embeddings = np.stack(validation_df['embeddings'])
    validation_embeddings = validation_embeddings.reshape(-1, validation_embeddings.shape[-1])

    #Get the anchor embeddings

    anchor_df = process_description_to_embeddings(anchor_df, 'description')
    anchor_embeddings = np.stack(anchor_df['embeddings'])

    target_list = ['risk analysis', 'risk identification', 'risk handling', 'opportunity management', 'risk tracking', 'risk management']

    f1_scores = []
    recall_scores = []
    balanced_accuracy_scores = []


    for column_name in target_list:
        anchor = anchor_df[anchor_df[column_name]==1]['embeddings'].iloc[0].numpy()

        #Get the positive and negative indices for the validation set
        val_pos_indices, val_neg_indices = get_positive_negative_indices(column_name, validation_df)

        #duplicate the anchor to match the batch size
        num_samples = len(validation_embeddings)
        anchor_embeddings = np.tile(anchor, (num_samples, 1))

        #Get prediction from the model
        prediction = model.predict([anchor_embeddings, validation_embeddings, validation_embeddings])

        # Extract embeddings from the dictionary
        predicted_values = prediction['loss'].values.numpy()

        val_pos_indices, val_neg_indices = get_positive_negative_indices(column_name, validation_df)

        # # Iterate through positive indices and print similarity scores to the anchor
        # print("Positive Examples:")
        # for idx, val_pos_idx in enumerate(val_pos_indices):
        #     pos_similarity_score = predicted_values[val_pos_idx]
        #     print(f"Positive Pair {idx+1} - Positive Similarity Score: {pos_similarity_score:.4f}")

        # # Iterate through negative indices and print similarity scores to the anchor
        # print("\nNegative Examples:")
        # for idx, val_neg_idx in enumerate(val_neg_indices):
        #     neg_similarity_score = predicted_values[val_neg_idx]
        #     print(f"Negative Pair {idx+1} - Negative Similarity Score: {neg_similarity_score:.4f}")

        # Create lists to store true labels and predicted labels
        true_labels = []
        predicted_labels = []

        threshold = 1.0

        # Iterate through the positive indices and classify pairs
        for val_pos_idx in val_pos_indices:
            true_labels.append(1)  # Positive label
            pos_similarity_score = predicted_values[val_pos_idx]
            predicted_labels.append(1 if pos_similarity_score >= threshold else 0)

        # Iterate through the negative indices and classify pairs
        for val_neg_idx in val_neg_indices:
            true_labels.append(0)  # Negative label
            neg_similarity_score = predicted_values[val_neg_idx]
            predicted_labels.append(1 if neg_similarity_score >= threshold else 0)

        # Plot the confusion matrix
        plot_confusion_matrix(true_labels, predicted_labels, normalize=False, figure_name = column_name, save_location=figure_path, validation=validation)

        # Calculate metrics
        f1 = f1_score(true_labels, predicted_labels)
        recall = recall_score(true_labels, predicted_labels)
        balanced_accuracy = balanced_accuracy_score(true_labels, predicted_labels)

        # Append the calculated metrics to the lists
        f1_scores.append(f1)
        recall_scores.append(recall)
        balanced_accuracy_scores.append(balanced_accuracy)

        # Write metrics to results.txt
        if validation:
            with open(f'{figure_path}/results_validation.txt', 'a') as results_file:
                results_file.write(column_name + ':\n')
                results_file.write(f'F1 Score: {f1:.4f}\n')
                results_file.write(f'Recall: {recall:.4f}\n')
                results_file.write(f'Balanced Accuracy: {balanced_accuracy:.4f}\n')
        else:
            with open(f'{figure_path}/results_test.txt', 'a') as results_file:
                results_file.write(column_name + ':\n')
                results_file.write(f'F1 Score: {f1:.4f}\n')
                results_file.write(f'Recall: {recall:.4f}\n')
                results_file.write(f'Balanced Accuracy: {balanced_accuracy:.4f}\n\n')

        # plot_confusion_matrix(true_labels, all_predictions, save_location=figure_path, figure_name = 'Risk_Identification_Validation', normalize=False)

    # Calculate average of each metric
    avg_f1 = sum(f1_scores) / len(f1_scores)
    avg_recall = sum(recall_scores) / len(recall_scores)
    avg_balanced_accuracy = sum(balanced_accuracy_scores) / len(balanced_accuracy_scores)

    # Print or write the average metrics to a results file
    with open(f'{figure_path}/results.txt', 'w') as avg_results_file:
        avg_results_file.write(f'Average F1 Score: {avg_f1:.4f}\n')
        avg_results_file.write(f'Average Recall: {avg_recall:.4f}\n')
        avg_results_file.write(f'Average Balanced Accuracy: {avg_balanced_accuracy:.4f}\n')


test_model = False
if test_model:
    test_df = read_excel_file(Data_location, testFileName)