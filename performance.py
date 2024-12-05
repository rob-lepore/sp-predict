from sklearn.metrics import precision_recall_curve, f1_score, matthews_corrcoef
from sklearn.metrics import matthews_corrcoef, f1_score, precision_score, recall_score, accuracy_score

import pandas as pd
import numpy as np
import sys

def best_threshold(validation_path):
    validation = pd.read_csv(validation_path, sep="\t", index_col=0, header=0)
    y_validation = validation["Class"]
    y_validation_scores = validation["Score"]

    precision, recall, thresholds = precision_recall_curve(y_validation, y_validation_scores)
    fscore = (2 * precision * recall) / (precision + recall)
    index = np.argmax(fscore)
    optimal_threshold = thresholds[index]
    return optimal_threshold

def performance(test_path, th):
    test = pd.read_csv(test_path, sep="\t", index_col=0, header=0)
    y_test_scores = test["Score"]
    y_pred_test = [int(t_s >= th) for t_s in y_test_scores]
    scores = [matthews_corrcoef(test["Class"], y_pred_test), 
              f1_score(test["Class"],y_pred_test),
              precision_score(test["Class"], y_pred_test),
              recall_score(test["Class"], y_pred_test),
              accuracy_score(test["Class"], y_pred_test)]
    return scores # matthews_corrcoef(test["Class"], y_pred_test)

if __name__ == "__main__":
    
    ths = []
    ps = []
    for i in range(1,6):
        th = best_threshold(f"vH_data/predictions_{i}_validation.tsv")
        p = performance(f"vH_data/predictions_{i}_test.tsv", th)
        ths.append(th)
        ps.append(p)
    print("Optimal thresholds: ", ths)
    print("Thresholds performances: ", ps)

    threshold = sum(ths)/len(ths)
    
    print("\nAverage threshold:", threshold)
    print(len(ps))
    for i in range(5): 
        values = []
        for p in ps:
            values.append(p[i])
        perf = sum(values)/len(values)
        err = np.std(values) / np.sqrt(len(values))
        print(f"Average performance: {perf} ±{err}")

    
    
    
    
    
   