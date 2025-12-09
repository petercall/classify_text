import pandas as pd
from functools import reduce
import json
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

"""
This file combines the probability vectors from a csv file that results from running the classify.py script. 
It combines them in one of two ways, depending on the value of the COMBINATION_TYPE variable:
    COMBINATION_TYPE = 'weighted' --> The probability vectors are combined using the original lengths of the text as weights
    COMBINATION_TYPE = 'normal' (or any other string besides 'weighted') --> A simple average is used to combine the probability vectors

Two things can then be done:
1. When COMBINE = True: Prints the combined weights of the different classes, or saves them to a csv file (depending on the values of PRINT_COMBINATION and SAVE_COMBINATION)

2. When CONVERGENCE = True: Tracks the difference between the probability vectors that result from combining successively more of the probability vectors. It stops when the difference between has been less than eps for more than patience number of successive iterations.
"""

#Hyperparameters------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
DATA_FILE = "../../../data/olmo/olmo_bos_1500_generations.csv"  #This is the filepath/filename.csv where your data is stored, which should be a csv file that was output from running the classify.py script.
COMBINATION_TYPE = 'weighted'   #This is either 'weighted' or any other string. If 'wieghted', it combines the probability vectors by weighting them according to the original lengths of the text. If any other string, then it combines the probability vectors via a simple average. 

#hyperparameters to combine the classification results
COMBINE = False              #This is either True or False and determines whether this seciton is skipped over (False), or whether the probability vectors are combined and either printed or saved (True).
PRINT_COMBINATION = False        #This is either True or False. If this is true it will print out the final classification scores.
SAVE_COMBINATION = "../outputs/classifications/olmo/olmo_bos_1500_generations_classification_results.csv"    #This can be False or a string filepath. If this is False it will not save the final combined probability vector. If it is a filepath then it should end in .csv and it will save the final combined probability vector as a csv file at that location.

#Hyperparameters to test the convergence of the classification results
CONVERGENCE = True    #This is either True or False and determines whether this section is skipped over (False) or whether convergence analysis is done and the graph saved and the convergence steps printed out.
eps = 0.0001          #The epsilon value to used for convergence
patience = 2          #If the difference between running probability vectors is less than eps for more than patience number of iterations, then it assumes convergence has occurred and stops.  
GRAPH_LOCATION = '../outputs/graphs/olma_bos_1500_convergence.png'      #This can be False or a string filepath. If it is False, no graph is saved and it only prints out the number of steps it took to converge. Otherwise it will save the graph of the convergence to the filepath and filename.filetype that you specify (doing .png is best). 
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


#Functions----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def combine(accum, new):
    #Extract the index and row from the input
    ind, new = new
    
    #Get the scores from the output column
    output_dict = json.loads(new["output"])
    scores = np.array(output_dict["scores"])

    #Multiply the scores by the original length
    if COMBINATION_TYPE == 'weighted':
        scores = scores * new["original_length"]
    
    #If index is 1, then accumm is the first row so we need to extract the scores
    if ind == 1:
        ind, accum = accum
        output_dict = json.loads(accum["output"])
        accum = np.array(output_dict["scores"])

    return accum + scores
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#Load in the dataset
df = pd.read_csv(DATA_FILE, header = 0)
df = df[~df["output"].isna()].reset_index(drop = True)

#Get the labels
labels = json.loads(df.at[0, "output"])["labels"]

#Run the combine function if specified
if COMBINE:    
    output = reduce(combine, df.iterrows())
    if COMBINATION_TYPE == 'weighted':
        classes = output / df["original_length"].sum()
    else:
        classes = output / df.shape[0]

    #Print out the different classes
    if PRINT_COMBINATION:
        for i in range(len(classes)):
            if len(labels[i]) < 6:
                print(f"{labels[i]}: \t\t\t{round(classes[i]*100, 2)}%")
            elif len(labels[i]) > 11:
                print(f"{labels[i]}: \t{round(classes[i]*100, 2)}%")
            else:
                print(f"{labels[i]}: \t\t{round(classes[i]*100, 2)}%")
                
    if SAVE_COMBINATION != False:
        #Create a new dataframe to hold the results
        result_df = pd.DataFrame({
            "label" : labels,
            "scores" : classes
        })
        
        #Save the dataframe to a csv file
        result_df.to_csv(SAVE_COMBINATION, index = False)
        print(f"Results saved to {SAVE_COMBINATION}")
        
if CONVERGENCE != False:
    #Get an original running_scores array and num_combined value of 0, and a running_weight of 0 if specified
    running_scores = np.zeros(len(labels))
    num_combined = 0
    if COMBINATION_TYPE == 'weighted':
        running_weight = 0
    
    #Loop through the dataframe and get the new vector each time and calculate its difference from the running vector
    num_stagnant = 0
    diff_list = []
    for i in range(df.shape[0]):
        #Calculate the current probabilities
        if i == 0:
            current_probs = np.zeros(len(labels))
        else:
            if COMBINATION_TYPE == 'weighted':
                current_probs = running_scores / running_weight
            else:
                current_probs = running_scores / num_combined
        
        #Get the current scores and weight if specified
        current_scores = np.array(json.loads(df.at[i, "output"])["scores"])
        if COMBINATION_TYPE == 'weighted':
            current_weight = df.at[i, "original_length"]
            
        #Increment the num_combined value up by 1
        num_combined += 1
            
        #Calculate the
        if COMBINATION_TYPE == 'weighted':
            running_weight += current_weight
            running_scores += current_scores * current_weight
            new_probs = running_scores / running_weight
        else:
            running_scores += current_scores
            new_probs = running_scores / num_combined
            
        #Get the difference between the current probability vector and the new probability vector, and add it to the diff_list
        diff = la.norm(current_probs - new_probs)
        diff_list.append(diff)
        
        #Increase num_stagnant if diff is less than eps
        if diff < eps:
            num_stagnant += 1
            
            #Break out of the loop if the difference has been small for 3 iterations in a row
            if num_stagnant > patience:
                break
        else:
            num_stagnant = 0
    
    if GRAPH_LOCATION != False:
        #Create the figure that   
        fig, ax = plt.subplots()
        ax.plot(range(len(diff_list)), diff_list)
        ax.set_xlabel("Iteration Number")
        ax.set_ylabel("Difference (2-norm) in Probability Vectors")
        ax.set_title("Convergence Analysis of Classification Task")

        # Add text bubble
        ax.text(
            0.95, 0.95,                      # X, Y position (as a fraction of axes)
            f"Convergence Occured on Iteration: {num_combined}",                      # Text
            transform=ax.transAxes,          # Use Axes-relative coordinates (0–1)
            fontsize=12,
            color="black",
            ha="right",                      # Align text to the right
            va="top",                        # Align text to the top
            bbox=dict(
                boxstyle="round,pad=0.4",    # Rounded bubble
                facecolor="lightblue",       # Bubble color
                edgecolor="black",           # Border color
                alpha=0.7                    # Transparency
            )
        )
        
        #Save the figure to the location specified above
        plt.savefig(GRAPH_LOCATION)
        
    #Print out how many steps it took to converge
    print(f"Convergence occurred on step: {num_combined}")