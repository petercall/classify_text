from datasets import load_dataset
import multiprocessing
from langdetect import detect
import random
import os

"""
The purpose of this file is to get a random sample of the dolma dataset.

In this file I do the following:
    1. Read in the v1_6-sample version of allenai/dolma
        1a. Print out the number of rows in the dataset
    2. Shuffle the dataset so it is in a random order (skipped if both orig_sample_size and final_sample size are 'all' because then the order is irrelavent)
    3. Sample orig_sample_size number of the rows if orig_sample_size != 'all'
    4. Filter out any rows that have original_length < min_length or whose language is not English      
        4a. Print out the number of rows sampled out an then number of rows remaining in the dataset  
    5. Then sample final_sample_size number of rows as long as there are at least that many left after the filter, and as long as final_sample_size != 'all'.
        5a. Print out the number of rows in the final dataset
    6. Create two new columns in the dataset: (1) short_text: A copy of the text that is shortened to be no longer than shortened_length number of characters, (2) original_length: The number of characters in the original text
    7. Save the dataset with columns (short_text, original_length) to the save_location as a csv file
"""

#Hyperparameters------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
orig_sample_size = 5300000        #This is the number of rows you will take on your first sample (just to pair down the dataset before proceeding) or 'all' if you do not want to take a first sample
final_sample_size = 5200000        #This is the final number of rows in the dataset that you will take, or 'all' if you want all of them
seed = 7                       #You can input an integer and it will use this seed, or you can input the string 'random' and it will choose a random seed in [1, 100]

min_length = 10          #Text values must be at least this long in order to be kept in the dataset
shortened_length = 500  #All texts will be concatenated to this number of characters, which should be enough to categorize the text but will make the categorization much faster computationally
save_location = '../../../data/olmo/dolma_v1_6_40_percent.csv'   #This is the location to save the csv file
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#Functions------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#This function filters out any texts that are not longer than min_length or are not English (this function is written assumming we will be using batch = True in the .filter() method)
def filter_out(batch):
    
    #Grab the texts
    texts = batch["text"]
    
    bools = []
    for text in texts:
        if not isinstance(text, str) or len(text) < min_length:
            bools.append(False)
            continue
        try:
            is_english = (detect(text) == 'en')
        except Exception:
            is_english = False
        bools.append(is_english)
        
    return bools

#This function adds two columns, one which counts the length of the original text, and one which contains the first shortened_length characters of the text (this function is written assumming we will be using batch = True in the .map() method)
def length_and_shortened_text(dict):
    #Get a list of the texts (and the associated id's)
    texts = dict["text"]
    ids = dict["id"]
    
    #Look through the text and add the length to the lengths list, raising an error if it is not of type string (like if it is NaN or something)
    lengths, short_texts = [], []
    for i, text in enumerate(texts):
        if isinstance(text, str):
            lengths.append(len(text))
            short_texts.append(text[:shortened_length])
        else:
            raise TypeError(f"The following instance was not of type string\nId: {ids[i]}\nType: {type(text)}\nContent: {text}")
    
    return {'short_text': short_texts, 'original_length' : lengths}

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------




#Change directory to the directory that this file is in
os.chdir(os.path.dirname(os.path.abspath(__file__)))

#Load in the data
data = load_dataset("allenai/dolma", name = "v1_6-sample", split = "train")

#Print the numbers of rows in the dataset
orig_length = data.num_rows
print(f"The dataset has the following number of rows: {orig_length}")

#Get a random seed if seed = 'random' and then shuffle the data
if seed == 'random':
    seed = random.randint(1,100)
if orig_sample_size != 'all' or final_sample_size != 'all':
    data = data.shuffle(seed=seed)

#Take a sample of size orig_sample_size of the overall dataset
if orig_sample_size != 'all':
    data = data.select(range(orig_sample_size))
new_length = data.num_rows
print(f"The first sample size has the following number of rows: {new_length}")

#Apply the filter which removes any rows that have fewer than min_length number of characters and are not English
data = data.filter(filter_out, batched=True, num_proc = multiprocessing.cpu_count() - 6)

#Print out the number of rows after the filter
filtered_length = data.num_rows
print()
print(f"The number of rows filtered out was: {new_length - filtered_length}")
print(f"The new number of rows is: {filtered_length}")

#Sample the final number of rows that you want 
if final_sample_size != 'all':
    if filtered_length >= final_sample_size:
        data = data.select(range(final_sample_size))
print(f"The final samples has the following number of rows: {data.num_rows}")

#Create two new columns which are the length of the original text and a shortened version of the text
data = data.map(length_and_shortened_text, batched=True, num_proc = multiprocessing.cpu_count() - 6, remove_columns = ["source", "added", "created", "text", "id"])

#Save the dataset
data.to_csv(save_location, index = False)
print(f"The dataset was saved to the following filepath: {os.path.abspath(save_location)}")