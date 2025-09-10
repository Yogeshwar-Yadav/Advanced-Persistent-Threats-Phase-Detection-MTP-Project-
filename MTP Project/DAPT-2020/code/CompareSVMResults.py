import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc

attack_types = ['portscan', 'sqli', 'lateral_movement', 'data_exfiltration']
plt.figure(figsize=(10, 8))
results_folder ="/home/yogeshwar/Yogesh-MTP/Results/csvfiles_AE/"

for attack in attack_types:
    fpr = np.loadtxt(f'/home/yogeshwar/Yogesh-MTP/Results/csvfiles_AE/custom-{attack}-fpr.csv')
    tpr = np.loadtxt(f'/home/yogeshwar/Yogesh-MTP/Results/csvfiles_AE/custom-{attack}-tpr.csv')
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{attack} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for One-Class SVM on DAPT 2020 Attacks')
plt.legend(loc='lower right')
plt.savefig(results_folder + '-roc.png')
plt.show()