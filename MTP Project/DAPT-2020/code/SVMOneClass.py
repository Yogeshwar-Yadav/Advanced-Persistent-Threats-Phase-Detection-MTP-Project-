import pandas as pd
import numpy as np
from sklearn import svm
import datagenerator_all as datagenerator
import joblib
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score


datasetType = input("Please enter datasetType from below:\n\tcicids2017\n\tcicids2018\n\tcustom\n\tunb15\n")
dataPath = input("Please enter the folder path from where the files need to picked up (without trailing slash)\n")
attackType = input("Please enter the attack type you want to predict from below:\n\treconnaissance\n\tfoothold_establishment\n\tlateral_movement\n\tdata_exfiltration\n")
results_folder ="/home/yogeshwar/Yogesh-MTP/Results_New/SVM/exfilteration/"
modelName = datasetType

dataset_train = datagenerator.loadDataset(dataPath + '/' + datasetType + "_normal.csv")
dataset_test = datagenerator.loadDataset(dataPath + '/' + datasetType + "_" + attackType + ".csv")

_nSamplesTrain = dataset_train.shape[0]
_nSamplesPred = dataset_test.shape[0]
_nColumns = dataset_train.shape[1]

X_train = datagenerator.getInput(datasetType, dataset_train, 0, _nSamplesTrain, _nColumns)

X_test = datagenerator.getInput(datasetType, dataset_test, 0, _nSamplesPred, _nColumns)
y_test = datagenerator.getLabelColumn(datasetType, dataset_test, 0, _nSamplesPred)

shouldTrain = input("Do you want to do training?[y/n]\n")
if shouldTrain == 'y':
    oneclass=svm.OneClassSVM(kernel='linear', gamma=0.000001, nu=0.10)
    model = oneclass.fit(X_train)
    joblib.dump(model, modelName + '.sav')
    print("Saved model to disk")
else:
    model = joblib.load(modelName + '.sav')
    print("Loaded model from disk")
    prediction = model.decision_function(X_test)
       
    data_n = pd.DataFrame(X_test)
    data_n = data_n.astype('float32')
    dist = np.zeros(_nSamplesPred)
    for i, x in enumerate(data_n.iloc[0:_nSamplesPred, :].values):
        dist[i] = np.linalg.norm(prediction[i])
        
    fpr,tpr,threshold = roc_curve(y_test, dist)
    roc_auc = auc(fpr, tpr)
    
    np.savetxt(results_folder + modelName+ '-' + attackType + '-fpr.csv', fpr, delimiter="\n")
    np.savetxt(results_folder + modelName+ '-' + attackType + '-tpr.csv', tpr, delimiter="\n")
    
    print("Generated fpr and tpr files")

          
    #----------------------------------------------------------
    #Plotting
    #----------------------------------------------------------
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(fpr, tpr, color='red', label='AUC = %0.2f)' % roc_auc)
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.plot([0, 1], [0, 1], color="navy", linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC One-Class SVM')
    plt.legend(loc='lower right')
    plt.show()
    plt.savefig(results_folder + modelName + '-' + attackType + '-roc.png')
    print("Generated ROC plot")
    #----------------------------------------------------------

# Precision-Recall calculation
precision, recall, pr_thresholds = precision_recall_curve(y_test, dist)
avg_precision = average_precision_score(y_test, dist)

# Save precision and recall arrays
np.savetxt(results_folder + modelName + '-' + attackType + '-precision.csv', precision, delimiter="\n")
np.savetxt(results_folder + modelName + '-' + attackType + '-recall.csv', recall, delimiter="\n")

print("Generated precision and recall files")

# Plotting PR Curve
plt.figure()
plt.plot(recall, precision, color='blue', label='Avg Precision = %0.2f' % avg_precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve - One-Class SVM')
plt.legend(loc='lower left')
plt.grid()
plt.savefig(results_folder + modelName + '-' + attackType + '-pr.png')
plt.show()

print("Generated PR plot")
