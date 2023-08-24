from sklearn.datasets import fetch_20newsgroups
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import string
import re
import contractions

twenty_data = fetch_20newsgroups(remove = ('headers', 'footers', 'quotes'), shuffle=True)

# Create a DataFrame containing the target labels and newsgroup names
target_df = pd.DataFrame({'target': twenty_data.target})
target_names = [twenty_data.target_names[i] for i in target_df['target']]
target_df['newsgroup'] = target_names

# Create one-hot encoded columns for each newsgroup
one_hot_encoded = pd.get_dummies(target_df['newsgroup'], columns=['newsgroup'])

# Concatenate the one-hot encoded columns with the original DataFrame
target_df = pd.concat([target_df, one_hot_encoded], axis=1)

# Drop the 'newsgroup' column as it's no longer needed
target_df.drop('newsgroup', axis=1, inplace=True)

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

processed_data = preprocess_column(twenty_data['data'])
processed_df = pd.DataFrame({'data': processed_data})
processed_twenty_data = pd.concat([target_df, processed_df], axis=1)

# Filter out rows with empty lists in the 'data' column
processed_twenty_data = processed_twenty_data[processed_twenty_data['data'].apply(lambda x: bool(x))]

# Reset the index after removing rows
processed_twenty_data.reset_index(drop=True, inplace=True)

# Count the number of words in each list in the 'data' column
word_count = processed_twenty_data['data'].apply(len)

# Filter out rows with less than 20 words
processed_twenty_data = processed_twenty_data[word_count >= 15]

# Reset the index after removing rows
processed_twenty_data.reset_index(drop=True, inplace=True)


# #Save dataframes to an excel file for further investigation
# save_csv_file = True
# if save_csv_file:
#     csv_file_path = "/remote_home/AML/CSCE_823_Project/Files/twenty_newsgroup_df.csv"

#     # Save the DataFrame to the Excel file
#     processed_twenty_data.to_csv(csv_file_path, index=False)
#     print(f"DataFrame saved to {csv_file_path}")

    #Save dataframes to an excel file for further investigation
save_excel_file = True
if save_excel_file:
    excel_file_path = "/remote_home/AML/CSCE_823_Project/Files/twenty_newsgroup_df.xlsx"

    # Save the DataFrame to the Excel file
    processed_twenty_data.to_excel(excel_file_path, index=False)
    print(f"DataFrame saved to {excel_file_path}")
