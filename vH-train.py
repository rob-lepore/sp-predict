import sys
import numpy as np
import pandas as pd
import math

if __name__ == "__main__":
    #  python3 vH-train.py train_positives.tsv vh_model.tsv
    train_path = sys.argv[1]
    out_path = sys.argv[2]
    
    swissprot_comp = pd.Series({
        'A': 8.25,
        'Q': 3.93,
        'L': 9.64,
        'S': 6.65,
        'R': 5.52,
        'E': 6.71,
        'K': 5.80,
        'T': 5.36,
        'N': 4.06,
        'G': 7.07,
        'M': 2.41,
        'W': 1.10,
        'D': 5.46,
        'H': 2.27,
        'F': 3.86,
        'Y': 2.92,
        'C': 1.38,
        'I': 5.91,
        'P': 4.74,
        'V': 6.85
    })/100
    aminos = swissprot_comp.index.to_list()
    
    # Get sequences of proteins in training set
    df = pd.read_csv(train_path, sep="\t", header=0, index_col=0)
    ids = df["ID"].to_list()
    seqs = {}
    with open("sets/positives.fasta") as fasta:
        for line in fasta:
            if line[0]==">":
                id = line[1:].rstrip()
            else:
                seqs[id] = line.rstrip()
    
    # Build matrix
    L = 15
    M = np.ones(shape=(L,20)) # pos: row, residue: col
    c = 0
    for id in ids:
        if math.isnan(df[df["ID"]==id]["SP_end"]): continue
        c+=1
        cleavage_site = int(df[df["ID"]==id]["SP_end"])
        begin = cleavage_site-13 # inclusive
        end = cleavage_site+2 # exclusive
        s = seqs[id][begin:end]
        # For each sequence, count the residue in each position
        for i in range(L):
            M[i][aminos.index(s[i])] += 1
    M /= 20 + c
    
    # Log-odd score against the swiss-prot distribution
    for i, residue in enumerate(aminos):
        M[:,i] /= swissprot_comp[residue]
    M=np.log2(M)
    M_df = pd.DataFrame(M, columns=aminos)
    print(M_df)
    
    M_df.to_csv(out_path, sep="\t")