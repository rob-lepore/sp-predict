import pandas as pd
import subprocess
from sklearn.metrics import matthews_corrcoef, f1_score, precision_score, recall_score, accuracy_score
import pickle, gzip
from svm_train import get_encoding
from sklearn import preprocessing


def get_metrics(y_true, y_pred):
    scores = [matthews_corrcoef(y_true, y_pred), 
              f1_score(y_true,y_pred),
              precision_score(y_true, y_pred),
              recall_score(y_true, y_pred),
              accuracy_score(y_true, y_pred)]
    scores_names = ["MCC", "F1", "Precision", "Recall", "Accuracy"]
    return scores, scores_names

if __name__ == "__main__":
    # VH
    vh_pred = pd.read_csv("vH_data/predictions_benchmark.tsv", sep="\t", header=0, index_col=0)
    threshold = 9.194
    vh_pred["Prediction"] = (vh_pred["Score"]>threshold).astype(int)
    vh_scores, score_names = get_metrics(vh_pred["Class"], vh_pred["Prediction"])
    vh_pred.to_csv("results/vH_predictions.tsv", sep="\t", header=True, index=True)
    vh_FN = vh_pred[(vh_pred["Prediction"] != vh_pred["Class"]) & (vh_pred["Class"] == 1)]
    vh_FP = vh_pred[(vh_pred["Prediction"] != vh_pred["Class"]) & (vh_pred["Class"] == 0)]
    vh_FN.to_csv("results/vH_FN.tsv", sep="\t", header=True, index=True)
    vh_FP.to_csv("results/vH_FP.tsv", sep="\t", header=True, index=True)
    
    # SVM
    feature_set = 13
    svc = pickle.load(gzip.open(f'svm_data/models/svm_{feature_set}.pkl.gz', 'r'))
    X_train, _ = get_encoding(f"svm_data/encodings_{feature_set}.tsv", "sets/train.tsv")
    scaler = preprocessing.MinMaxScaler().fit(X_train)
    
    X_test, y_test = get_encoding(f"svm_data/encodings_{feature_set}.tsv", "sets/test.tsv")
    X_test_scaled = scaler.transform(X_test)
    y_test_pred = svc.predict(X_test_scaled)
    
    svc_preds = {"Class": {}, "Prediction":{}}
    for i, id in enumerate(X_test.index):
        svc_preds["Class"][id] = y_test[i]
        svc_preds["Prediction"][id] = y_test_pred[i]
    svc_preds = pd.DataFrame(svc_preds)
    
    svc_FN = svc_preds[(svc_preds["Prediction"] != svc_preds["Class"]) & (svc_preds["Class"] == 1)]
    svc_FP = svc_preds[(svc_preds["Prediction"] != svc_preds["Class"]) & (svc_preds["Class"] == 0)]
    
    svc_preds.to_csv("results/svm_predictions.tsv", sep="\t", header=True, index=True)
    svc_FN.to_csv("results/svm_FN.tsv", sep="\t", header=True, index=True)
    svc_FP.to_csv("results/svm_FP.tsv", sep="\t", header=True, index=True)