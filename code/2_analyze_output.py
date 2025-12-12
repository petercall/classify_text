import pandas as pd
from functools import reduce
import json
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import os

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
FILENAME = "phi_2000_generations_temp_1.5"   #This can be used to quickly adjust all 3 locations that the filename shows up
DATA_FILE = f"../../../data/olmo/{FILENAME}.csv"  #This is the filepath/filename.csv where your data is stored, which should be a csv file that was output from running the classify.py script (which means it has an "output" column).
COMBINATION_TYPE = 'weighted'   #This is either 'weighted' or any other string. If 'weighted', it combines the probability vectors by weighting them according to the original lengths of the text. If any other string, then it combines the probability vectors via a simple average. 
WEIGHT_COLUMN = 'original_length'   #This is ONLY used if COMBINATION_TYPE = 'weighted'. In this case, this gives the column name that contains the integer weights.
TOP_K = 3                           #This is how many of the top classes should be used in the combination. You can put 'all' to use all classes, or an integer to use that many of the top classes.

#hyperparameters to combine the classification results
COMBINE = True              #This is either True or False and determines whether this seciton is skipped over (False), or whether the probability vectors are combined and either printed or saved (True).
PRINT_COMBINATION = False        #This is either True or False. If this is true it will print out the final classification scores.
SAVE_COMBINATION = f"../outputs/classifications/olmo/{FILENAME}_classification_results.csv"    #This can be False or a string filepath. If this is False it will not save the final combined probability vector. If it is a filepath then it should end in .csv and it will save the final combined probability vector as a csv file at that location.

#Hyperparameters to test the convergence of the classification results
CONVERGENCE = True    #This is either True or False and determines whether this section is skipped over (False) or whether convergence analysis is done and the graph saved and the convergence steps printed out.
eps = 1e-4            #The epsilon value to used for convergence
patience = 3          #If the difference between running probability vectors is less than eps for more than patience number of iterations, then it assumes convergence has occurred and stops.  
GRAPH_LOCATION = f'../outputs/graphs/{FILENAME}_convergence.png'      #This can be False or a string filepath. If it is False, no graph is saved and it only prints out the number of steps it took to converge. Otherwise it will save the graph of the convergence to the filepath and filename.filetype that you specify (doing .png is best). 
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


#Functions----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def combine(accum, new):
    #Extract the index and row from the input
    _, new = new
    
    #Get the scores from the output column
    output_dict = json.loads(new["output"])
    scores = np.array(output_dict["scores"])
    
    #Get the ordering of the largest scores and set all but the TOP_K to 0
    if TOP_K != 'all':
        ordering = np.argsort(scores)[::-1]
        scores[ordering[TOP_K:]] = 0

    #Multiply the scores by the original length
    if COMBINATION_TYPE == 'weighted':
        scores = scores * new[WEIGHT_COLUMN]

    return accum + scores
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------




#Change to the current working directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

#Load in the dataset
df = pd.read_csv(DATA_FILE, header = 0)
df = df[~df["output"].isna()].reset_index(drop = True)

#Get the labels and count how many there are
labels = json.loads(df.at[0, "output"])["labels"]
num_labels = len(labels)

#Run the combine function if specified
if COMBINE:    
    #Run the combine function which sums the top_k probabilities of each vector and weights them if specified.
    output = reduce(combine, df.iterrows(), np.zeros(num_labels))
    
    #Divide by the sum of output in order to normalize the output so that it is a probability vector (i.e. so it sums to 1)
    classes = output / output.sum()

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
    #Setup the initial values before running through the loop
    running_scores = np.zeros(num_labels)
    num_stagnant = 0
    diff_list = []
    
    #Loop through the dataframe and get the new vector each time and calculate its difference from the running vector
    for i in range(df.shape[0]):
        #Calculate the current probabilities
        if i == 0:
            current_probs = np.zeros(num_labels)
        else:
            current_probs = running_scores / running_scores.sum()
        
        #Get the current scores
        current_scores = np.array(json.loads(df.at[i, "output"])["scores"])
        
        #Weight the current scores if specified, and mask out any but the TOP_K scores if specified
        if COMBINATION_TYPE == 'weighted':
            current_scores *= df.at[i, "original_length"]
        if TOP_K != 'all':
            ordering = np.argsort(current_scores)[::-1]
            current_scores[ordering[TOP_K:]] = 0
        
        #Get the new running_scores value and the new_probs
        running_scores += current_scores
        new_probs = running_scores / running_scores.sum()
            
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
        #Calculate the number of iterations in the convergence test
        num_iterations = len(diff_list)
        
        #Create the figure that   
        fig, ax = plt.subplots()
        ax.plot(range(num_iterations), diff_list)
        ax.set_xlabel("Iteration Number")
        ax.set_ylabel("Difference (2-norm) in Probability Vectors")
        ax.set_title("Convergence Analysis of Classification Task")

        # Add text bubble
        ax.text(
            0.95, 0.95,                      # X, Y position (as a fraction of axes)
            f"Convergence Occurred on Iteration: {num_iterations}",                      # Text
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
    print(f"Convergence occurred on step: {num_iterations}")