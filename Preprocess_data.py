import json
import os
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
from sklearn.model_selection import train_test_split
import sys
from sklearn.preprocessing import LabelEncoder
import pickle

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import string
import re
import contractions

data_list = []

# Directory where JSON files are located
json_directory = "/remote_home/AML/Avolve_data/cipub"

# Loop through each file in the directory
for filename in os.listdir(json_directory):
    # Check if the file is a JSON file
    if filename.endswith(".json"):
        file_path = os.path.join(json_directory, filename)
        with open(file_path, "r") as file:
            json_data = json.load(file)
            title = json_data.get("title", "")
            description = json_data.get("description", "")

            topics = []
            topics_data = json_data.get("topics", {})
            for topic_id, topic_list in topics_data.items():
                for topic in topic_list:
                    topics.append(topic)

            contentType = json_data.get("contentType", "")
            contentSubType = json_data.get("contentSubType", "")

            # Extract keywords
            keywords = json_data.get("keywords", [])

            if description and keywords:
                data_list.append({
                "file_name": filename,
                "title": title,
                "description": description,
                "topics": topics,
                "keywords": keywords,
                "contentType": contentType,
                "contentSubType": contentSubType,
                })

# Create a pandas DataFrame from the data list
df = pd.DataFrame(data_list)

# Convert 'keywords' column entries to lowercase
df['keywords'] = df['keywords'].apply(lambda keywords: [kw.lower() for kw in keywords])


def split_dataframe_by_keywords(dataframe, keyword_list):
    matching_keywords = []
    non_matching_keywords = []

    for index, row in dataframe.iterrows():
        matched_keywords = [keyword for keyword in row["keywords"] if keyword in keyword_list]
        if matched_keywords:
            matching_keywords.append({"title": row["title"], "description": row["description"],  "keywords": matched_keywords})
        else:
            non_matching_keywords.append({"title": row["title"], "description": row["description"], "keywords": row["keywords"]})


    matching_df = pd.DataFrame(matching_keywords)
    non_matching_df = pd.DataFrame(non_matching_keywords)

    # Reset indices for the DataFrames
    matching_df = matching_df.reset_index(drop=True)
    non_matching_df = non_matching_df.reset_index(drop=True)

    return matching_df, non_matching_df

target_topics = ['opportunity', 'risk', 'risk management', 'risk analysis', 'risk handling',
'risk tracking', 'opportunity management', 'risk identification']

non_training_df, training_df = split_dataframe_by_keywords(df, target_topics)

target_training_topics = ['cyber', 'innovation', 'acquisition', 'cybersecurity', 'toc', 'intelligence', 'theory of constraints',
'tesseract', 'intelligence analysis', 'python data science', 'training', 'mbse', 'digital engineering', 'digital', 'afit',
'cyber security']

training_df, extra_df = split_dataframe_by_keywords(training_df, target_training_topics)

# Because the keywords aren't large, we're going to use one-hot encoding to differentiate the keywords
def create_one_hot_encoded_columns(dataframe, keywords):
    for keyword in keywords:
        dataframe[keyword] = dataframe['keywords'].apply(lambda keywords_list: 1 if keyword in keywords_list else 0)
    dataframe.drop(columns=['keywords'], inplace=True)
    return dataframe

# Now one hot encode the training and non_training datasets
non_training_df = create_one_hot_encoded_columns(non_training_df, target_topics)
training_df = create_one_hot_encoded_columns(training_df, target_training_topics)

#Now perform pre-processing on the description row
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

#This function assists with preprocessing the description column.
def preprocess_column(column):
    def expand_contractions(text):
        expanded_text = contractions.fix(text)
        return expanded_text
    
    def clean_text(text):
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove non-ASCII characters
        text = re.sub(r'[{}]'.format(re.escape(string.punctuation + ',')), ' ', text)  # Remove punctuation and commas
        text = re.sub(r'\s+', ' ', text)  # Replace consecutive spaces with a single space
        return text.strip()  # Trim leading and trailing whitespace
    
    def preprocess_text(text):
        return clean_text(expand_contractions(text))
    
    # Tokenize words and preprocess
    tokenized = [word_tokenize(sentence) for sentence in column]
    cleaned = [[preprocess_text(word) for word in sentence if preprocess_text(word)] for sentence in tokenized]
    
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    without_stopwords = [[word for word in sentence if word.lower() not in stop_words] for sentence in cleaned]
    
    # Normalize words: convert to lowercase and apply lemmatization
    lemmatizer = WordNetLemmatizer()
    normalized = [[lemmatizer.lemmatize(word.lower()) for word in sentence] for sentence in without_stopwords]
    
    return normalized

#normalize the data in the description column for both datasets
preprocessed_training = preprocess_column(training_df['description'])
training_df['description'] = preprocessed_training

preprocessed_non_training = preprocess_column(non_training_df['description'])
non_training_df['description'] = preprocessed_non_training

#Save dataframes to an excel file for further investigation
save_csv_file = False
if save_csv_file:
    csv_file_path = "/remote_home/AML/CSCE_823_Project/Files/non_training_df.csv"

    # Save the DataFrame to the Excel file
    non_training_df.to_csv(csv_file_path, index=False)
    print(f"DataFrame saved to {csv_file_path}")

if save_csv_file:
    csv_file_path = "/remote_home/AML/CSCE_823_Project/Files/training_df.csv"

    # Save the DataFrame to the Excel file
    training_df.to_csv(csv_file_path, index=False)
    print(f"DataFrame saved to {csv_file_path}")

# We need to create a custom dataset that evenly divides the non_training data based on an even distribution of labels



