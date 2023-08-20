import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import string
import re
import contractions


Data_location = "/remote_home/AML/CSCE_823_Project/Data"

anchor_data_unprocessed = "anchor_data_df_unprocessed.xlsx"

def read_excel_file(file_path, file_name, sheet_name=0):
    # Combine the file path and file name
    excel_file_path = f"{file_path}/{file_name}"
    
    # Read the Excel file and create a DataFrame
    df = pd.read_excel(excel_file_path, sheet_name=sheet_name)
    
    return df

anchor_df = read_excel_file(Data_location, anchor_data_unprocessed)


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
anchor_df_description = preprocess_column(anchor_df['description'])
anchor_df['description'] = anchor_df_description

print(anchor_df['description'])

save_csv_file = True
if save_csv_file:
    csv_file_path = "/remote_home/AML/CSCE_823_Project/Files/anchor_data_df_processed.csv"

    # Save the DataFrame to the Excel file
    anchor_df.to_csv(csv_file_path, index=False)
    print(f"DataFrame saved to {csv_file_path}")