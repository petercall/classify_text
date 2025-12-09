from datasets import load_dataset
import multiprocessing
from langdetect import detect
import random

"""
The purpose of this file is to get a random sample of the dolma dataset

In this file I do the following:
    1. Read in the v1_6-sample version of allenai/dolma
        1a. Print out the number of rows in the dataset
    2. Sample orig_sample_size number of the rows
    3. Create a new column named original_length that is the number of characters in the text
        2a. Throw an error if any of the texts are not strings (i.e. if they are NaN)
    4. Filter out any that have original_length < min_length or whose language is not English    
    5. Then sample final_sample_size number of rows as long as there are at least that many left after the filter.
        5a. Print out the number of rows in the final dataset
    6. Create a new column called short_text which is the shortened version of the text, clamping the text down to shortened_length number of characters
    7. Save the dataset with columns (id, short_text, original_length) to the save_location as a csv file
"""

#Hyperparameters------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
orig_sample_size = 50000     #This is the number of rows you will take on your first sample, just to pair down the dataset before proceeding
final_sample_size = 7000    #This is the final number of rows in the dataset that you will take
seed = 17                     #You can input an integer and it will use this seed, or you can input the string 'random' and it will choose a random seed in [1, 100]

min_length = 10          #Text values must be at least this long in order to be kept in the dataset
shortened_length = 500  #All texts will be concatenated to this number of characters, which should be enough to categorize the text but will make the categorization much faster computationally
save_location = '../../../../data/olmo/dolma_v1_6_subset2.csv'   #This is the location to save the csv file
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




#Load in the data
data = load_dataset("allenai/dolma", name = "v1_6-sample", split = "train")

#Print the numbers of rows in the dataset
orig_length = data.num_rows
print(f"The original number of rows is: {orig_length}")

#Get a random seed if seed = 'random'
if seed == 'random':
    seed = random.randint(1,100)

#Take a sample of size orig_sample_size of the overall dataset
data = data.shuffle(seed=seed).select(range(orig_sample_size))

#Apply the filter which removes any rows that have fewer than min_length number of characters and are not English
data = data.filter(filter_out, batched=True, num_proc = multiprocessing.cpu_count() - 6)

#Print out the number of rows after the filter
new_length = data.num_rows
print()
print(f"The number of rows filtered out was: {orig_sample_size - new_length}")
print(f"The new number of rows is: {new_length}")

#Sample the final number of rows that you want 
if data.num_rows >= final_sample_size:
    data = data.shuffle(seed=seed).select(range(final_sample_size))
print(f"The final dataset has {data.num_rows} number of rows")

#Create two new columns which are the length of the original text and a shortened version of the text
data = data.map(length_and_shortened_text, batched=True, num_proc = multiprocessing.cpu_count() - 6, remove_columns = ["source", "added", "created", "text"])

#Save the dataset
data.to_csv(save_location, index = False)