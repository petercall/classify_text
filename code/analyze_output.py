import pandas as pd
from functools import reduce
import json
import numpy as np
import scipy.linalg as la

#Hyperparameters----------------------------------------------------------------------------------------------------------------------------------------------------------------
#Data hyperparameters
# DATA_FILE = "../../../data/olmo/olmo_bos_1500_generations.csv"
DATA_FILE = '../../../data/olmo/dolma_classification_subset.csv'

#hyperparameters to combine the classification results
COMBINE = False        #If this is true it will combine the classification results
WEIGHTED_COMBINE = False      #If this is true it will weight the classification scores by the original length of the text
PRINT_COMBINE = False        #If this is true it will print out the final classification scores
SAVE_LOCATION_COMBINE = "../outputs/classifications/olmo/olmo_bos_1500_generations_classification_results.csv"    #If this is False it will not save them. If it is a location it will save them at that location

#Hyperparameters to test the convergence of the classification results
CONVERGENCE = True   #If this is true it will test the convergence of the classification results
WEIGHTED_CONVERGENCE = True     #If this is true it will weight the classification scores by the original length of the text
eps = 0.00001          #The epsilon value to use for convergence
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------


#Functions--------------------------------------------------------------------------------------------------------------------------------------------------------------------
def combine(accum, new):
    #Extract the index and row from the input
    ind, new = new
    
    #Get the scores from the output column
    output_dict = json.loads(new["output"])
    scores = np.array(output_dict["scores"])

    #Multiply the scores by the original length
    if WEIGHTED_COMBINE:
        scores = scores * new["original_length"]
    
    #If index is 1, then accumm is the first row so we need to extract the scores
    if ind == 1:
        ind, accum = accum
        output_dict = json.loads(accum["output"])
        accum = np.array(output_dict["scores"])

    return accum + scores
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------

#Load in the dataset
df = pd.read_csv(DATA_FILE, header = 0)
df = df[~df["output"].isna()].reset_index(drop = True)

#Get the labels
labels = json.loads(df.at[0, "output"])["labels"]

#Run the combine function if specified
if COMBINE:    
    output = reduce(combine, df.iterrows())
    if WEIGHTED_COMBINE:
        classes = output / df["original_length"].sum()
    else:
        classes = output / df.shape[0]

    #Print out the different classes
    if PRINT_COMBINE:
        for i in range(len(classes)):
            if len(labels[i]) < 6:
                print(f"{labels[i]}: \t\t\t{round(classes[i]*100, 2)}%")
            elif len(labels[i]) > 11:
                print(f"{labels[i]}: \t{round(classes[i]*100, 2)}%")
            else:
                print(f"{labels[i]}: \t\t{round(classes[i]*100, 2)}%")
                
    if SAVE_LOCATION_COMBINE != False:
        #Create a new dataframe to hold the results
        result_df = pd.DataFrame({
            "label" : labels,
            "scores" : classes
        })
        
        #Save the dataframe to a csv file
        result_df.to_csv(SAVE_LOCATION_COMBINE, index = False)
        print(f"Results saved to {SAVE_LOCATION_COMBINE}")
        
if CONVERGENCE:
    #Get an original running_scores array and num_combined value of 1, and a running_weight if specified
    running_scores = np.zeros(len(labels))
    num_combined = 0
    if WEIGHTED_CONVERGENCE:
        running_weight = 0
    
    #Loop through the dataframe and get the new vector each time and calculate its difference from the running vector
    num_stagnant = 0
    for i in range(df.shape[0]):
        #Calculate the current probabilities
        if i == 0:
            current_probs = np.zeros(len(labels))
        else:
            if WEIGHTED_CONVERGENCE:
                current_probs = running_scores / running_weight
            else:
                current_probs = running_scores / num_combined
        
        #Get the current scores and weight if specified
        current_scores = np.array(json.loads(df.at[i, "output"])["scores"])
        if WEIGHTED_CONVERGENCE:
            current_weight = df.at[i, "original_length"]
            
        #Increment the num_combined value up by 1
        num_combined += 1
            
        #Calculate the
        if WEIGHTED_CONVERGENCE:
            running_weight += current_weight
            running_scores += current_scores * current_weight
            new_probs = running_scores / running_weight
        else:
            running_scores += current_scores
            new_probs = running_scores / num_combined
            
        #Get the difference vector
        diff = la.norm(current_probs - new_probs)
        
        #Increase num_stagnant if diff is less than eps
        if diff < eps:
            num_stagnant += 1
            
            #Break out of the loop if the difference has been small for 3 iterations in a row
            if num_stagnant == 3:
                break
        else:
            num_stagnant = 0
        
    print(f"Convergence occurred on step: {num_combined}")