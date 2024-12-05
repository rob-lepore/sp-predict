from svm_train import get_encoding, train_SVM
from sklearn import svm
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, matthews_corrcoef, precision_score, recall_score, f1_score, confusion_matrix
from sklearn import preprocessing
import math

def make_encoding_3c(encoding_path, index, set_df):
    encodings = pd.read_csv(encoding_path, sep="\t", index_col=0, header=0)
    y=[]
    for id in set_df["ID"]:
        if math.isnan(set_df[set_df["ID"]==id]["SP_end"]):
            y.append(int(set_df[set_df["ID"]==id]["TM"])) # 0: Neg, 1: TM
        else:
            y.append(2)
    encodings["Label"] = y
    encodings.to_csv(f"svm_data/multiclass/encodings_{index}.tsv", sep="\t", header=True, index=True)
    
    
def train_with_gs(encoding_path,train_path,validation_path,test_path, grid):
    X, y = get_encoding(encoding_path, train_path)
    X_val, y_val = get_encoding(encoding_path, validation_path)
    X_test, y_test = get_encoding(encoding_path, test_path)
    
    scaler = preprocessing.MinMaxScaler().fit(X)

    # Grid search
    models = []
    scores = []
    for c, gamma in grid:
        # Train on the training set
        model = svm.SVC(C=c, gamma=gamma, kernel='rbf')        
        model.fit(scaler.transform(X), y)
        models.append(model)
        # Evaluate on the validation set
        y_pred = model.predict(scaler.transform(X_val))
        scores.append(performances(y_val, y_pred)[0])
    best_model = models[np.argmax(scores)]
    
    # Evaluate best model on the testing set
    y_test_pred = best_model.predict(scaler.transform(X_test))
    perf = performances(y_test, y_test_pred)
    model_scores = [np.max(scores), *perf]
    scores_names = ["MCC (val)","MCC", "F1", "Precision", "Recall", "Accuracy"]
    return best_model, model_scores, scores_names

def performances(y_true, y_pred):
    cm_3 = confusion_matrix(y_true, y_pred)
    cm = np.zeros((2,2))
    cm[0,0] = cm_3[0,0] + cm_3[0,1] + cm_3[1,0] + cm_3[1,1]
    cm[0,1] = cm_3[0,2] + cm_3[1,2]
    cm[1,0] = cm_3[2,0] + cm_3[2,1]
    cm[1,1] = cm_3[2,2]
    
    # Extracting values from the confusion matrix
    TP = cm[1, 1]
    FP = cm[1, 0]
    FN = cm[0, 1]
    TN = cm[0, 0]

    mcc = (TP * TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (TP + TN) / (TP + FP + FN + TN)
    return mcc, f1_score, precision, recall, accuracy

if __name__ == "__main__":
     
    for f_set in range(1,17):   
        print("\n\nFeature set: ",f_set) 
        grid = [(c, g) for c in [1,2,4,8] for g in [0.5,1,2,'scale']]
        
        cv_results = []
        for n in range(1,6):
            model, scores, score_names = train_with_gs(f"svm_data/multiclass/encodings_{f_set}.tsv", f"folds/fold_{n}_train.tsv", f"folds/fold_{n}_validation.tsv", f"folds/fold_{n}_test.tsv", grid)
            cv_results.append([model.get_params()['C'], model.get_params()['gamma'], *scores])
        results_df = pd.DataFrame(cv_results, columns=["C", "gamma", *score_names], index=[f"CV run #{n}" for n in range(1,6)])
        #print(results_df)
        print(results_df.iloc[:,2:].mean())
        print(results_df.iloc[:,2:].std()/np.sqrt(5))
        
        c_value_counts = results_df["C"].value_counts()
        c_most_frequent = c_value_counts[c_value_counts == c_value_counts.max()].index.min()
        gamma_value_counts = results_df["gamma"].value_counts()
        gamma_most_frequent = gamma_value_counts[gamma_value_counts == gamma_value_counts.max()].index
        if "scale" in gamma_most_frequent and gamma_most_frequent.shape[0]>1:
            gamma_most_frequent = gamma_most_frequent.drop("scale")
        gamma_most_frequent = gamma_most_frequent.min()
        
        hparams = [c_most_frequent, gamma_most_frequent]
        print("\nSelected C: ", hparams[0])
        print("Selected gamma: ", hparams[1]) 
        
    
    out_file = open("results/analysis/MC_FP_rates.txt", "w")
    
    for f_set, hparams in [(5,[2,1]), (16,[8,'scale'])]:
        
        svc = train_SVM(f"svm_data/multiclass/encodings_{f_set}.tsv","sets/train.tsv",hparams[0], hparams[1])
        X_train, _ = get_encoding(f"svm_data/multiclass/encodings_{f_set}.tsv", "sets/train.tsv")
        scaler = preprocessing.MinMaxScaler().fit(X_train)
        
        X_test, y_test = get_encoding(f"svm_data/multiclass/encodings_{f_set}.tsv", "sets/test.tsv")
        X_test_scaled = scaler.transform(X_test)
        y_test_pred = svc.predict(X_test_scaled)
        
        print(performances(y_test, y_test_pred), file=out_file) 

        print("Feature set:", f_set, file=out_file)
        predictions = pd.DataFrame({"Class": y_test, "Prediction":y_test_pred}, index=X_test.index)
            
        # FALSE POSITIVES
        
        FP = predictions[(predictions["Prediction"] != predictions["Class"]) & (predictions["Prediction"] == 2)]
        N = predictions[(predictions["Class"] < 2)]
        n_FP = FP.shape[0]
        n_N = N.shape[0]
        print("False positives over negatives: %.2f%%" % (100*n_FP / n_N), file=out_file)
        
        FP_tm = FP[FP["Class"] == 1]
        N_tm = predictions[(predictions["Class"] == 1)]
        print("False positives with TM over negatives with TM: %.2f%%" % (100*(FP_tm.shape[0])/N_tm.shape[0]), file=out_file)
    out_file.close()