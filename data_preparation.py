import subprocess
import pandas as pd

def cluster(fasta_file_name, out_prefix, out_folder="clustering", si_threshold = 0.3, cov_threshold = 0.4):
    command = [
        "mmseqs", "easy-cluster", fasta_file_name, f"{out_folder}/{out_prefix}", f"{out_folder}/tmp",
        "--min-seq-id", str(si_threshold), "-c", str(cov_threshold), "--cov-mode", "0", "--cluster-mode", "1"
    ]
    subprocess.run(command)#, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    
def get_representatives(cluster_fasta_name, original_tsv_name, out_tsv_name):
    reps = {}
    with open(cluster_fasta_name) as file:
        for line in file:
            if line[0] != ">": continue
            reps[line[1:].rstrip()] = 1
    with open(original_tsv_name) as file:
        with open(out_tsv_name, "w") as out:
            for line in file:
                if line.split("\t")[0] in reps:
                    print(line, file=out, end="")
    return reps

def split_train_test(df: pd.DataFrame, frac):
    df_train = df.sample(frac=frac, random_state=42)
    df_test = df.drop(df_train.index)
    return df_train, df_test
    
def make_folds(df: pd.DataFrame, n_folds):
    folds = []
    fold_size = int((1/n_folds)*df.shape[0])
    for i in range(n_folds-1):
        fold = df.sample(n=fold_size, random_state=42)
        folds.append(fold)
        df.drop(fold.index, inplace=True)
    folds.append(df)
    return folds
    
    
if __name__ == "__main__":
    print("Clustering positives")
    cluster("positives.fasta", "cluster_positives")
    print("Clustering negatives")
    cluster("negatives.fasta", "cluster_negatives")
    
    pos_reps = get_representatives("clustering/cluster_positives_rep_seq.fasta", "positives.tsv", "positives_reps.tsv")
    print(f"{len(pos_reps)} positive representative sequences")
    neg_reps = get_representatives("clustering/cluster_negatives_rep_seq.fasta", "negatives.tsv", "negatives_reps.tsv")
    print(f"{len(neg_reps)} negative representative sequences")
    
    # Split train/test
    df_pos_refs = pd.read_csv("positives_reps.tsv", sep="\t", header=None, names=["ID", "Organism", "Kingdom", "Length", "SP_end"])    
    df_pos_train, df_pos_test = split_train_test(df_pos_refs, frac=0.8)
    df_neg_refs = pd.read_csv("negatives_reps.tsv", sep="\t", header=None, names=["ID", "Organism", "Kingdom", "Length", "TM"])    
    df_neg_train, df_neg_test = split_train_test(df_neg_refs, frac=0.8)
    print(f"Training set: {df_pos_train.shape[0]} positives, {df_neg_train.shape[0]} negatives ({df_pos_train.shape[0] + df_neg_train.shape[0]} total)")
    print(f"Testing set: {df_pos_test.shape[0]} positives, {df_neg_test.shape[0]} negatives ({df_pos_test.shape[0] + df_neg_test.shape[0]} total)")
    
    # Fold separation
    n_folds = 5
    pos_folds = make_folds(df_pos_train, n_folds)
    neg_folds = make_folds(df_neg_train, n_folds)
    
    # Set fold column
    for i in range(n_folds):
        pos_folds[i]["Fold"] = i+1
        neg_folds[i]["Fold"] = i+1
    
    df_pos = pd.concat(pos_folds, ignore_index=True)
    df_neg = pd.concat(neg_folds, ignore_index=True)
    
    df_pos.to_csv("train_positives.tsv", sep="\t")
    df_neg.to_csv("train_negatives.tsv", sep="\t")
    df_pos_test.to_csv("test_positives.tsv", sep="\t")
    df_neg_test.to_csv("test_negatives.tsv", sep="\t")
    pd.concat([df_pos, df_neg], ignore_index=True).to_csv("train.tsv", sep="\t")
    pd.concat([df_pos_test, df_neg_test], ignore_index=True).to_csv("test.tsv", sep="\t")
    
    