from datasets import load_dataset
import pandas as pd
import os


"""
The purpose of this file is to get a random sample of the finemath dataset

In this file I do the following:
    1. Read in the finemath-4plus version of HuggingFaceTB/finemath with streaming=True
    2. Grab just sample_size number of rows
    3. Create a new column named original_length that is the number of characters in the text and a shortened version of the text called short_text with shortened_length number of characters
    4. Save the dataset with columns (short_text, original_length) to the save_location as a csv file
"""

#Hyperparameters------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
sample_size = 5000    #This is the final number of rows in the dataset that you will take
shortened_length = 500  #All texts will be concatenated to this number of characters, which should be enough to categorize the text but will make the categorization much faster computationally
save_location = '../../../data/olmo/math_dataset.csv'   #This is the location to save the csv file
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#Functions------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#This is the function that grabs sample_size number of rows from the dataset
def get_subset(sample_size=sample_size):
    ds_stream = load_dataset("HuggingFaceTB/finemath", "finemath-4plus", split="train", streaming=True)
    subset = []
    for i, example in enumerate(ds_stream):
        subset.append(example['text'])
        if i + 1 >= sample_size:
            break
    return subset

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------




#Change directory to the directory that this file is in
os.chdir(os.path.dirname(os.path.abspath(__file__)))

#Load in the data
subset = list(get_subset())

#Turn the data into a pandas dataframe
data = pd.DataFrame({"text" : subset})

#Add the original length and shortened text to the dataframe
data["original_length"] = data["text"].str.len()
data["short_text"] = data["text"].str[:shortened_length]

#Grab just the short_text and orignial_length columns
data = data[["short_text", "original_length"]]

#Save the csv file to the location specified
data.to_csv(save_location, index=False)
