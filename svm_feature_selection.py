from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.feature_selection import SelectFromModel, VarianceThreshold, SelectPercentile, chi2
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
#import xgboost as xgb
import pandas as pd
from sklearn.metrics import accuracy_score, matthews_corrcoef, precision_score, recall_score, f1_score
from sklearn import preprocessing
from sklearn import svm
from svm_train import get_encoding
import numpy as np


def cross_validate(f, selector):
    X, y = get_encoding(encoding, f"folds/fold_{f}_train.tsv")
    scaler = preprocessing.MinMaxScaler().fit(X)
    X_scaled = scaler.transform(X)
    
    selector.fit(X_scaled,y)
    
    grid = [(c, g) for c in [1,2,4,8] for g in [0.5,1,2,'scale']]
    models = []
    scores = []
    for c, gamma in grid:
        # Train on the training set
        model = svm.SVC(C=c, gamma=gamma, kernel='rbf')
        X_selected = selector.transform(X_scaled)
        model.fit(X_selected,y)
        
        # Evaluate on the validation set
        X_val, y_val = get_encoding(encoding, f"folds/fold_{f}_validation.tsv") 
        X_val_trans = selector.transform(scaler.transform(X_val))
        y_pred = model.predict(X_val_trans)
        
        models.append(model)
        scores.append(matthews_corrcoef(y_val, y_pred))
    best = np.argmax(scores)
    
    X_test, y_test = get_encoding(encoding, f"folds/fold_{f}_test.tsv") 
    X_test_trans = selector.transform(scaler.transform(X_test))
    
    y_test_pred = models[best].predict(X_test_trans)
    
    model_scores = [scores[best], 
              matthews_corrcoef(y_test, y_test_pred), 
              f1_score(y_test,y_test_pred),
              precision_score(y_test, y_test_pred),
              recall_score(y_test, y_test_pred),
              accuracy_score(y_test, y_test_pred)]
    scores_names = ["MCC (val)","MCC", "F1", "Precision", "Recall", "Accuracy"]
    
    return models[best], (model_scores, scores_names) 

if __name__ == "__main__":
    encoding = "svm_data/encodings_16.tsv"
    
    pd.set_option("display.max_columns", None)
    pd.set_option("precision", 3)
    
    classifiers = [
        svm.LinearSVC(dual=False),
        svm.LinearSVC(C=0.01, max_iter=5000, penalty="l1", dual=False),
        GradientBoostingClassifier(), 
        RandomForestClassifier(),
        ExtraTreesClassifier(),
        DecisionTreeClassifier(),
        #KNeighborsClassifier(n_neighbors=3)
        #xgb.XGBClassifier(tree_method="hist")
    ]
    
    selectors = [
        VarianceThreshold(threshold=0.01),
        SelectPercentile(chi2, percentile=75),
        #SequentialFeatureSelector(KNeighborsClassifier(n_neighbors=3))
    ]
    selectors.extend([SelectFromModel(clf) for clf in classifiers])
    
    for selector in selectors:
        print(selector)
        
        results = []
        for f in range(1,6):
            
            model, (scores, score_names) = cross_validate(f, selector)
            
            C = model.get_params()["C"]
            gamma = model.get_params()["gamma"]
            results.append([C, gamma, sum(selector.get_support()), *scores])
        results_df = pd.DataFrame(results, columns=["C", "gamma", "# Feat", *score_names], index=[f"CV run {n}" for n in range(1,6)])
        print(results_df.iloc[:,:5])
        
        c_value_counts = results_df["C"].value_counts()
        c_most_frequent = c_value_counts[c_value_counts == c_value_counts.max()].index.min()
        gamma_value_counts = results_df["gamma"].value_counts()
        gamma_most_frequent = gamma_value_counts[gamma_value_counts == gamma_value_counts.max()].index
        if "scale" in gamma_most_frequent and len(gamma_most_frequent)>1:
            gamma_most_frequent = gamma_most_frequent.drop("scale")
        gamma_most_frequent = gamma_most_frequent.min()
        
        print("\nAverages:")
        avgs = results_df.iloc[:,4:].mean().astype(object)
        avgs["C"]=c_most_frequent
        avgs["gamma"]=gamma_most_frequent #if gamma_most_frequent != "scale" else -1
        print(avgs)
        print("\n\n")