import sys
import pandas as pd
from svm_train import train_with_gs, train_SVM
import pickle, gzip
import re

if __name__ == "__main__":
    encoding_path = sys.argv[1]
    model_out_path = sys.argv[2]
    
    pd.set_option("display.max_columns", None)
    pd.set_option("precision", 3)
    
    grid = [(c, g) for c in [1,2,4,8] for g in [0.5,1,2,'scale']]
    
    cv_results = []
    for n in range(1,6):
        model, scores, score_names = train_with_gs(encoding_path, f"folds/fold_{n}_train.tsv", f"folds/fold_{n}_validation.tsv", f"folds/fold_{n}_test.tsv", grid)
        cv_results.append([model.get_params()['C'], model.get_params()['gamma'], *scores])
    results_df = pd.DataFrame(cv_results, columns=["C", "gamma", *score_names], index=[f"CV run #{n}" for n in range(1,6)])
    print(results_df)
    
    c_value_counts = results_df["C"].value_counts()
    c_most_frequent = c_value_counts[c_value_counts == c_value_counts.max()].index.min()
    gamma_value_counts = results_df["gamma"].value_counts()
    gamma_most_frequent = gamma_value_counts[gamma_value_counts == gamma_value_counts.max()].index
    if "scale" in gamma_most_frequent:
        gamma_most_frequent = gamma_most_frequent.drop("scale")
    gamma_most_frequent = gamma_most_frequent.min()
    
    hparams = [c_most_frequent, gamma_most_frequent]
    print("\nSelected C: ", hparams[0])
    print("Selected gamma: ", hparams[1])
    
    # Print summary to file
    sum_file = open("svm_data/cv_summary.tsv", "a")
    feat_set = re.search(r'\d+', encoding_path).group()
    mean_values_str = "\t".join(f"{val:.3f}" for val in results_df[score_names[1:]].mean())
    summary = f"{feat_set}\t{hparams[0]}\t{hparams[1]}\t{mean_values_str}"
    print(summary, file=sum_file)
    sum_file.close()
    
    print("Training on the complete training set")
    model_train = train_SVM(encoding_path,"sets/train.tsv", hparams[0], hparams[1])
    pickle.dump(model, gzip.open(model_out_path, 'w'))
    
