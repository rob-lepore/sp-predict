import sys
from sklearn import svm
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, matthews_corrcoef, precision_score, recall_score, f1_score
from sklearn import preprocessing


def get_encoding(encoding_path, set_path):
    encodings = pd.read_csv(encoding_path, sep="\t", index_col=0, header=0)
    set_df = pd.read_csv(set_path, sep="\t", index_col=0, header=0)
    set_encodings = encodings.loc[set_df["ID"]]
    labels = set_encodings["Label"]
    encodings = set_encodings.drop(columns=["Label"])
    return encodings, labels

def train_with_gs(encoding_path,train_path,validation_path,test_path, grid):    
    # Get encodings and labels
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
        scores.append(matthews_corrcoef(y_val, y_pred))
    best_model = models[np.argmax(scores)]
    
    # Evaluate best model on the testing set
    y_test_pred = best_model.predict(scaler.transform(X_test))
    model_scores = [np.max(scores), 
              matthews_corrcoef(y_test, y_test_pred), 
              f1_score(y_test,y_test_pred),
              precision_score(y_test, y_test_pred),
              recall_score(y_test, y_test_pred),
              accuracy_score(y_test, y_test_pred)]
    scores_names = ["MCC (val)","MCC", "F1", "Precision", "Recall", "Accuracy"]
    return best_model, model_scores, scores_names
    
def train_SVM(encoding_path, train_path, C, gamma):
    X, y = get_encoding(encoding_path, train_path)
    scaler = preprocessing.MinMaxScaler().fit(X)
    model = svm.SVC(C=C, gamma=gamma, kernel='rbf')
    model.fit(scaler.transform(X), y)
    return model