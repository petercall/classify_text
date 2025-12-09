from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset
import pandas as pd
from collections.abc import Sequence
from transformers import pipeline
import numpy as np
from tqdm import tqdm
import os
import json

"""
This file takes a csv file that contains a text column (which you specify using the COL_OF_INT varaible) and classifies each row of the column on the TARGET_CLASSES which you specify, using the Hugging Face model specified by the MODEL_NAME variable.
It adds a new column to the csv file with name CLASSIFICATION_COL.
Each entry of that column is a json string of a python dictionary with key, value pairs of:
    'labels' --> The TARGET_CLASSES list sorted alphabetically
    'scores' --> A list of the scores that the model output of the similarity of that row's text with the associated label from the TARGET_CLASS list
"""


#Hyperparameters--------------------------------------------------------------------------------------------------------------------------------------------------------
DATA_FILE = "../../../data/olmo/dolma_classification_subset.csv"
COL_OF_INT = "0"         #This is the column it will be doing the classifying based on
CLASSIFICATION_COL = "output"  #This is the column where it will put the subject that had the highest classification score
NUM_TO_DO = "all"               #If this is "all" it will go from the start_position to the end of the file. If it is a number, it will do that many
START_POSITION = "first nan"    #If this is "first nan" it will start with the first nan found in CLASSIFICATION_COL. If a number, it will start with the index that equals that number
TARGET_CLASSES = [
    "Mathematics",
    "Physical Sciences",
    "Biological Sciences",
    "Social and Behavioral Sciences",
    "Engineering Sciences",
    "Computer Science and AI",
    "Medicine and Health",
    "Business and Economics",
    "Humanities and Arts",
    "Law and Government"
]
MODEL_NAME = "facebook/bart-large-mnli"
BATCH_SIZE = 64
SAVE_FREQ = 490     #This is the number of data points after which it will save (NOT the number of batches)
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------



#Change the location to the current working directory, to avoid all confusion
os.chdir(os.path.dirname(os.path.abspath(__file__)))

#Create the dataset class
class MyData(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        return self.data[i]

#Download the model and tokenizer and input them into a pipeline
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
pipe = pipeline("zero-shot-classification", model = model, tokenizer = tokenizer, device = "cuda")

#Download the data
data = pd.read_csv(DATA_FILE, header = 0)

#Create the new column if needed
if CLASSIFICATION_COL not in data.columns:
    data[CLASSIFICATION_COL] = np.nan

#Get the number of starting position
if START_POSITION == "first nan":
    START_POSITION = data.shape[0] - data[CLASSIFICATION_COL].isna().sum()
    
#Define the number to do
if NUM_TO_DO == "all":
    NUM_TO_DO = data.shape[0] - START_POSITION
    
#Load the data into a series and then a dataset
series = data[COL_OF_INT].iloc[START_POSITION : START_POSITION + NUM_TO_DO].reset_index(drop = True)
series = series.dropna().reset_index(drop = True)
dataset = MyData(series)

#Iterate over your dataset and fill the labels Series with the model output
try:
    for i, output in enumerate(tqdm(pipe(dataset, TARGET_CLASSES, batch_size = BATCH_SIZE), total = len(series))):
        #Sort the labels and scores to be alphabetical
        indices = np.argsort(output["labels"])
        labels = np.array(output["labels"])[indices]
        labels = list(labels)
        scores = np.array(output["scores"])[indices]
        scores = list(scores)
        
        #Put the labels and scores into a json string and set that into the dataframe
        my_dict = {"labels" : labels, "scores" : scores}
        my_string = json.dumps(my_dict)
        data.loc[START_POSITION + i, CLASSIFICATION_COL] = my_string

        if i % SAVE_FREQ == 0:
            data.to_csv(DATA_FILE, index = False)                
except Exception as e:
    print(f"exception found: {e}")
finally:
    data.to_csv(DATA_FILE, index = False)             