import sys
import numpy as np
import pandas as pd
import math

if __name__ == "__main__":
    # python3 vH-predict.py vh_model.tsv test.tsv <(cat positives.fasta negatives.fasta) predictions.txt
    model_path = sys.argv[1]
    test_tsv_path = sys.argv[2]
    test_fasta_path = sys.argv[3]
    out_path = sys.argv[4]
    
    M = pd.read_csv(model_path, sep="\t", header=0, index_col=0)
    
    df = pd.read_csv(test_tsv_path, sep="\t", header=0, index_col=0)
    ids = df["ID"].to_list()
    seqs = {}
    with open(test_fasta_path) as fasta:
        for line in fasta:
            if line[0]==">":
                id = line[1:].rstrip()
            else:
                seqs[id] = line.rstrip()[:90]

    L = M.shape[0]
    
    results = {}
    for id in ids:
        seq = seqs[id]
        scores = []
        for j in range(len(seq)-L+1):
            score = 0
            for i in range(j,j+L):
                if seq[i] in ['X', 'U', 'B']: continue
                score += M.loc[i-j,seq[i]]
            scores.append(score)
        results[id] = (max(scores), int(not math.isnan(df[df["ID"]==id]["SP_end"])))
    pd.DataFrame.from_dict(results, columns=["Score", "Class"], orient='index').to_csv(out_path, sep="\t")