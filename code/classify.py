from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset
import pandas as pd
from collections.abc import Sequence
from transformers import pipeline
import numpy as np
from tqdm import tqdm
import os

#Hyperparameters--------------------------------------------------------------------------------------------------------------------------------------------------------
DATA_FILE = "../../data/subjects/train.csv"
COL_OF_INT = "question"         #This is the column it will be doing the classifying based on
CLASSIFICATION_COL = "subject"  #This is the column where it will put the subject that had the highest classification score
NUM_TO_DO = "all"               #If this is "all" it will go from the start_position to the end of the file. If it is a number, it will do that many
START_POSITION = "first nan"    #If this is "first nan" it will start with the first nan found in CLASSIFICATION_COL. If a number, it will start with the index that equals that number


# TARGET_CLASSES = ['abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge', 'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics', 'college_medicine', 'college_physics', 'computer_security', 'conceptual_physics', 'econometrics', 'electrical_engineering', 'elementary_mathematics', 'formal_logic', 'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science', 'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics', 'high_school_macroeconomics', 'high_school_mathematics', 'high_school_microeconomics', 'high_school_physics', 'high_school_psychology', 'high_school_statistics', 'high_school_us_history', 'high_school_world_history', 'human_aging', 'human_sexuality', 'international_law', 'jurisprudence', 'logical_fallacies', 'machine_learning', 'management', 'marketing', 'medical_genetics', 'miscellaneous', 'moral_disputes', 'moral_scenarios', 'nutrition', 'philosophy', 'prehistory', 'professional_accounting', 'professional_law', 'professional_medicine', 'professional_psychology', 'public_relations', 'security_studies', 'sociology', 'us_foreign_policy', 'virology', 'world_religions']
TARGET_CLASSES = ["mathematics", "logic", "physics", "chemistry", "biology", "medicine", "history", "social sciences", "computer science", "business", "law", "philosophy"]
MODEL_NAME = "facebook/bart-large-mnli"
BATCH_SIZE = 64
SAVE_FREQ = 150     #This is the number of data points after which it will save (NOT the number of batches)
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------



#input--------------------------------------------------------------------------------------------------------------------------------------------------------------
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
dataset = MyData(series)

#Iterate over your dataset and fill the labels Series with the model output
try:
    for i, output in enumerate(tqdm(pipe(dataset, TARGET_CLASSES, batch_size = BATCH_SIZE), total = len(series))):
        data.loc[START_POSITION + i, CLASSIFICATION_COL] = output["labels"][0]

        if i % SAVE_FREQ == 0:
            data.to_csv(DATA_FILE, index = False)                
except:
    print("exception found")
finally:
    data.to_csv(DATA_FILE, index = False)             
        
