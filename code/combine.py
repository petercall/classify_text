import pandas as pd
from functools import reduce
import json
import numpy as np


#Hyperparameters----------------------------------------------------------------------------------------------------------------------------------------------------------------
DATA_FILE = "../../../data/olmo/dolma_classification_subset.csv"
PRINT = True        #If this is true it will print out the final classification scores
SAVE_LOCATION = "../outputs/classifications/dolma_classification_results.csv"    #If this is False it will not save them. If it is a location it will save them at that location
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------


#Load in the dataset
df = pd.read_csv(DATA_FILE, header = 0)
# df = df[:10]

#Get the labels
labels = json.loads(df.at[0, "output"])["labels"]

#Create the reduce function
def combine(accum, new):
    #Extract the index and row from the input
    ind, new = new
    
    #Get the scores from the output column
    output_dict = json.loads(new["output"])
    scores = np.array(output_dict["scores"])

    #Multiply the scores by the original length
    scores = scores * new["original_length"]
    
    #If index is 1, then accumm is the first row so we need to extract the scores
    if ind == 1:
        ind, accum = accum
        output_dict = json.loads(accum["output"])
        accum = np.array(output_dict["scores"])

    return accum + scores
    
output = reduce(combine, df.iterrows())
classes = output / df["original_length"].sum()

#Print out the different classes
if PRINT:
    for i in range(len(classes)):
        if len(labels[i]) < 6:
            print(f"{labels[i]}: \t\t\t{round(classes[i].item()*100, 2)}%")
        elif len(labels[i]) > 11:
            print(f"{labels[i]}: \t{round(classes[i].item()*100, 2)}%")
        else:
            print(f"{labels[i]}: \t\t{round(classes[i].item()*100, 2)}%")
            
if SAVE_LOCATION != False:
    #Create a new dataframe to hold the results
    result_df = pd.DataFrame({
        "label" : labels,
        "scores" : classes
    })
    
    #Save the dataframe to a csv file
    result_df.to_csv(SAVE_LOCATION, index = False)
    print(f"Results saved to {SAVE_LOCATION}")