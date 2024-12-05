import sys
import pandas as pd

if __name__ == "__main__":
    train_path = sys.argv[1]
    train = pd.read_csv(train_path, sep="\t", header=0, index_col=0)
    
    for f in range(1,6):
        fold_test = train[train["Fold"] == f]
        fold_validation = train[train["Fold"] == (f+1)%5+1]
        fold_train = train.drop(fold_test.index, inplace=False)
        fold_train.drop(fold_validation.index, inplace=True)
        
        fold_test.to_csv(f"folds/fold_{f}_test.tsv", sep="\t")
        fold_validation.to_csv(f"folds/fold_{f}_validation.tsv", sep="\t")
        fold_train.to_csv(f"folds/fold_{f}_train.tsv", sep="\t")
        
        