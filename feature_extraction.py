import sys
import pandas as pd
import numpy as np
import math
import itertools
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio.SeqUtils import ProtParamData

zhao_london_scale = {
    'A': 0.380,  
    'R': -2.570, 
    'N': -1.620, 
    'D': -3.270, 
    'C': -0.300, 
    'Q': -1.840, 
    'E': -2.900, 
    'G': -0.190, 
    'H': -1.440, 
    'I': 1.970,  
    'L': 1.820,  
    'K': -3.460, 
    'M': 1.400,  
    'F': 1.980,  
    'P': -1.440, 
    'S': -0.530, 
    'T': -0.320, 
    'W': 1.530,  
    'Y': 0.490,  
    'V': 1.460   
}

charged_scale = {
    'A': 0,
    'R': 1,
    'N': 0,
    'D': 1,
    'C': 0,
    'Q': 0,
    'E': 1,
    'G': 0,
    'H': 1,
    'I': 0,
    'L': 0,
    'K': 1,
    'M': 0,
    'F': 0,
    'P': 0,
    'S': 0,
    'T': 0,
    'W': 0,
    'Y': 0,
    'V': 0
}



def get_sequences(fasta_path):
    seqs = {}
    with open(fasta_path) as fasta:
        for line in fasta:
            if line[0]==">":
                id = line[1:].rstrip()
            else:
                seqs[id] = line.rstrip()
    return seqs

def get_aa_composition(df: pd.DataFrame, sequences):
    comps = {}
    for id in df["ID"]:
        X = ProteinAnalysis(sequences[id][:22])
        freqs = X.get_amino_acids_percent()
        comps[id] = []
        for aa in ['A', 'Q', 'L', 'S', 'R', 'E', 'K', 'T', 'N', 'G', 'M', 'W', 'D', 'H', 'F', 'Y', 'C', 'I', 'P', 'V']:
            comps[id].append(freqs[aa])
    return comps

def get_feature(sequence, window_size, scale, funcs, padding=False):
    if padding:
        d = window_size // 2
        sequence = "X"*d + sequence + "X"*d
    analysed_seq = ProteinAnalysis(sequence)
    scl = analysed_seq.protein_scale(scale, window_size)
    res = [f(scl) for f in funcs]
    return res

def encode(df, sequences, scale, window_size, stats):
    values = {}
    for id in df["ID"]:
        values[id] = get_feature(sequences[id][:40], window_size, scale, stats, padding=False)
    return values


if __name__ == "__main__":
    fasta = sys.argv[1]
    train = pd.read_csv("sets/train.tsv",sep="\t", header=0, index_col=0)
    test = pd.read_csv("sets/test.tsv",sep="\t", header=0, index_col=0)
    all = pd.concat([train, test], ignore_index=True)

    sequences = get_sequences(fasta)
    aa_compositions = get_aa_composition(all, sequences)
    hydrophobicities = encode(all, sequences, ProtParamData.kd, 5, [lambda x: max(x), lambda x: np.mean(x)])
    helix_propensities = encode(all, sequences, ProtParamData.fs, 7, [lambda x: max(x), lambda x: np.mean(x)])
    charged = encode(all, sequences, charged_scale, 3, [lambda x: max(x), lambda x: np.argmax(x)/40])
    tm_propensities = encode(all, sequences, zhao_london_scale, 7, [lambda x: max(x), lambda x: np.mean(x)])
    
    features_all_names = ["Composition", "Hydrophobicity", "Alpha-helix propensity", "Charge", "Transmembrane helix propensity"]
    features_all = [aa_compositions, hydrophobicities, helix_propensities, charged, tm_propensities]
    feat_comb = []
    for r in range(1, len(features_all)):
        feat_comb.extend(itertools.combinations(list(range(1,len(features_all))), r))
    feat_comb = [[0,*list(comb)] for comb in feat_comb]
    feat_comb.insert(0,[0])
    print(f"{len(feat_comb)} combinations of features")
    
    comb_file = open("svm_data/feature_sets.txt", "w")
    for n, comb in enumerate(feat_comb):
        print(f"{n+1}/{len(feat_comb)}")
        print(f"Feature set {n+1}:", ", ".join([features_all_names[i] for i in comb]), file=comb_file)
        X = []
        y = []
        for id in all["ID"]:
            encoding = []
            for i in comb:
                encoding.extend(features_all[i][id])
            X.append(encoding)
            y.append(int(not math.isnan(all[all["ID"]==id]["SP_end"])))
        df = pd.DataFrame(X, index=all["ID"])
        df["Label"] = y
        df.to_csv(f"svm_data/encodings_{n+1}.tsv", sep="\t", header=True, index=True)
        
    comb_file.close()