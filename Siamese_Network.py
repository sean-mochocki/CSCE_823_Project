import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

Data_location = "/remote_home/AML/CSCE_823_Project/Data"
testFileName = "test_df.xlsx"
trainingFileName = "training_df.xlsx"
validationFileName = "validation_df.xlsx"

def read_excel_file(file_path, file_name, sheet_name=0):
    # Combine the file path and file name
    excel_file_path = f"{file_path}/{file_name}"
    
    # Read the Excel file and create a DataFrame
    df = pd.read_excel(excel_file_path, sheet_name=sheet_name)
    
    return df

test_df = read_excel_file(Data_location, testFileName)
training_df = read_excel_file(Data_location, trainingFileName)
validation_df = read_excel_file(Data_location, validationFileName)

print(test_df)