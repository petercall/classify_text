import pandas as pd

"""
This file is used to compare two different classification results, each of which is a csv file that was saved via the SAVE_COMBINATION variable in the analyze_output.py script.
It prints out a table of scores with the scores from the first csv file, the scores from the second csv file, and then the difference of the two.
"""


#Hyperparameters------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
filepath_1 = '../outputs/classifications/olmo/dolma_classification_results.csv'   #This is a csv file that was output from the analyze_output.py file
name_1 = 'Data'     #This is the name that will appear above the scores for this csv file
filepath_2 = '../outputs/classifications/olmo/olmo_bos_1500_generations_classification_results.csv'   #This is a second csv file that was output from the analyze_output.py file
name_2 = 'Model'    #This is the name that will appear above the scores for this csv file
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



#Read in the classification results of the two different datasets
df1 = pd.read_csv(filepath_1, header = 0)
df2 = pd.read_csv(filepath_2, header = 0)

#Rename the scores and label features so that they aren't the same when we combine them
df1.rename({"scores": "scores_1", "label": "labels_1"}, axis = 1, inplace = True)
df2.rename({"scores": "scores_2", "label": "labels_2"}, axis = 1, inplace = True)

#Combine the two datsets
combined = pd.concat([df1, df2], axis = 1, ignore_index = False)
combined["diff"] = combined["scores_1"] - combined["scores_2"]

print()
print(f"Label\t\t\t{name_1} Score\t{name_2} Score\tDifference")
for i in range(combined.shape[0]):
    if len(combined.at[i, "labels_1"]) < 6:
        print(f"{combined.at[i, 'labels_1']}: \t\t\t{round(combined.at[i, 'scores_1']*100, 2)}%\t\t{round(combined.at[i, 'scores_2']*100, 2)}%\t\t{round(combined.at[i, 'diff']*100, 2)}%")
    elif len(combined.at[i, 'labels_1']) > 11:
        print(f"{combined.at[i, 'labels_1']}: \t{round(combined.at[i, 'scores_1']*100, 2)}%\t\t{round(combined.at[i, 'scores_2']*100, 2)}%\t\t{round(combined.at[i, 'diff']*100, 2)}%")
    else:
        print(f"{combined.at[i, 'labels_1']}: \t\t{round(combined.at[i, 'scores_data']*100, 2)}%\t\t{round(combined.at[i, 'scores_2']*100, 2)}%\t\t{round(combined.at[i, 'diff']*100, 2)}%")
print()