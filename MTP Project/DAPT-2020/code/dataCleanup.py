#This script cleans up the file the way you need. Usually, the files need to cleaned from NaN and Infinity values. 
#So this script attempts to replace these values only. 
#You will be prompted with selections to choose from for replacing these values.
#You can run this script from any folder. Just make sure to either give absolute or relative path to the file to be cleaned.
#The cleaned files will be created beside the original files.
#The bigger the size of the file, the longer it takes to clean. So, please be patient..
import pandas as pd
import numpy as np

datasetDirectory = '/home/yogeshwar/Yogesh-MTP/csv/'
fileToClean = input("Please enter the filename to clean with out the extension:\n")
hasHeader = input("Please enter 0 or 1 from below:\n\tNo Header -> 0\n\tHas Header -> 1\n")
replaceInfinity = input("Please enter -1, 0 or 1 from below for replacing Infinity with\n \t\t\t\t\t\t-1 -> -1\n (used for cicids2017)\t 0 -> 0\n twice the max value for the feature -> 1\n")
replaceNan = input("Please enter -1, 0 or 1 from below for replacing nan with\n \t\t\t\t\t\t-1 -> -1\n (used for cicids2017)\t 0 -> 0\n")

if hasHeader == "0": #No Header
    headerRow = None
else:
    headerRow = 0

dataset = pd.read_csv(datasetDirectory + fileToClean + ".csv", header=headerRow, encoding="ISO-8859-1")
X = dataset.iloc[:, :].values

nColumns = len(X[0])
print("\n\n\nThe bigger the size of the file, the longer it takes to clean. Please be patient.....")
print("*********************************************")
if replaceInfinity == "1":
    print("Replacing Infinity with the max value of the feature and nan with " + str(replaceNan) + ".....")    
    
    print("\tRetrieving records without Infinity values to calculate max values.....")
    strippedDataset = []
    for i in range(len(X)):
        if 'nan' not in X[i] and 'Infinity' not in X[i]:
            strippedDataset.append(X[i])
    strippedDataset = np.array(strippedDataset)        
    
    print("\tCalculating max values of each feature to replace Infinity......")
    maxValues = []
    stringColumns = []
    infinityColumns = []
    for j in range(nColumns):
        try:
            maxValue = np.amax(X[:, j])
            if maxValue == 'Infinity':
                maxValue = float(np.amax(strippedDataset[:, j])) * 2
                print("\tInfinity values in column with index " + str(j) + " will be replaced by " + str(maxValue))
            maxValue = float(maxValue)            
        except: 
            maxValue = 0
            stringColumns.append(j)
        finally: 
            maxValues.append(maxValue)                    
    for i in range(len(X)):
        for j in range(nColumns):
            if str.lower(str(X[i, j])) == "infinity":                
                X[i, j] = float(maxValues[j])
            elif str.lower(str(X[i, j])) == 'nan':
                X[i, j] = replaceNan
else:
    print("\n\n\nReplacing Infinity and nan with " + str(replaceInfinity) + ", " + str(replaceNan) + " respectively.....")                          
    for i in range(len(X)):
        for j in range(nColumns):
            if str.lower(str(X[i, j])) == 'infinity':
                X[i, j] = replaceInfinity
            elif str.lower(str(X[i, j])) == 'nan':
                X[i, j] = replaceNan

            
print("Creating cleaned file.....")
fileCleaned = datasetDirectory + "Cleaned/" + fileToClean + ".csv"
np.savetxt(fileCleaned, np.array(X), delimiter=',', fmt="%s")
print("*********************************************")
print("Cleaned file " + fileCleaned + " has been created")            