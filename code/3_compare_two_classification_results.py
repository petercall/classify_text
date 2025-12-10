import pandas as pd
import os

"""
This file is used to compare two different classification results, each of which is a csv file that was saved via the SAVE_COMBINATION variable in the analyze_output.py script.
It prints out a table of scores with the scores from the first csv file, the scores from the second csv file, and then the difference of the two.
"""


#Hyperparameters------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
file_name1 = "dolma_v1_6_subset1"
name_1 = 'Dolma'     #This is the name that will appear above the scores for this csv file

file_name2 = "phi_2000_generations_temp_1.5"
name_2 = 'Olmo'    #This is the name that will appear above the scores for this csv file
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------




#Change to the current working directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

#Read in the classification results of the two different datasets
filepath_1 = f"../outputs/classifications/olmo/{file_name1}_classification_results.csv"
filepath_2 = f"../outputs/classifications/olmo/{file_name2}_classification_results.csv"
df1 = pd.read_csv(filepath_1, header = 0)
df2 = pd.read_csv(filepath_2, header = 0)

#Rename the scores and label features so that they aren't the same when we combine them
df1.rename({"scores": "scores_1", "label": "labels_1"}, axis = 1, inplace = True)
df2.rename({"scores": "scores_2", "label": "labels_2"}, axis = 1, inplace = True)

#Combine the two datsets
combined = pd.concat([df1, df2], axis = 1, ignore_index = False)
combined["diff"] = combined["scores_1"] - combined["scores_2"]

#Calculate the absolute value of the difference category and sort by that
combined["abs"] = combined["diff"].abs()
combined = combined.sort_values("abs", ascending = False).reset_index(drop = True)

print()
print(f"Label\t\t\t\t{name_1} Score\t{name_2} Score\tDifference")
for i in range(combined.shape[0]):
    #Figure out how much whitespace to give each line
    if len(combined.at[i, "labels_1"]) > 28:
        whitespace = ''
    elif len(combined.at[i, 'labels_1']) > 21:
        whitespace = '\t'
    elif len(combined.at[i, 'labels_1']) > 15:
        whitespace = '\t\t'
    else:
        whitespace = '\t\t\t'
    
    #Print out the results of that line        
    print(f"{combined.at[i, 'labels_1']}: {whitespace}{round(combined.at[i, 'scores_1']*100, 2)}%\t\t{round(combined.at[i, 'scores_2']*100, 2)}%\t\t{round(combined.at[i, 'diff']*100, 2)}%")
print()