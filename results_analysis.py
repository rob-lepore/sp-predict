import pandas as pd
import numpy as np
from feature_extraction import get_sequences, get_aa_composition
import matplotlib.pyplot as plt
import seaborn as sns

def get_total_composition(compositions: dict):
    comp_tot = pd.DataFrame.from_dict(compositions,orient='index')
    comp_tot.columns = ['A', 'Q', 'L', 'S', 'R', 'E', 'K', 'T', 'N', 'G', 'M', 'W', 'D', 'H', 'F', 'Y', 'C', 'I', 'P', 'V']
    comp_tot = comp_tot.reindex(["G","A","V","P","L","I","M","F","W","Y","S","T","C","N","Q","H","D","E","K","R"], axis=1)
    
    comp_tot = comp_tot * 22
    comp_tot = comp_tot.sum()/(len(compositions)*22)
    return comp_tot

if __name__ == "__main__":
    
    sns.set_style("whitegrid")
    sns.color_palette("bright")
    
    out_file = open("results/analysis/FP_rates.txt", "w")

    # SVM
    print("Support Vector Machine:", file=out_file)
    predictions = pd.read_csv("results/svm_predictions.tsv", sep="\t", index_col=0, header=0)
    test_set = pd.read_csv("sets/test.tsv", sep="\t", index_col=0, header=0)
    train_set = pd.read_csv("sets/train.tsv", sep="\t", index_col=0, header=0)
        
    # FALSE POSITIVES
    
    FP = predictions[(predictions["Prediction"] != predictions["Class"]) & (predictions["Class"] == 0)]
    N = test_set[test_set["SP_end"].isna()]
    n_FP = FP.shape[0]
    n_N = N.shape[0]
    print("False positives over negatives: %.2f%%" % (100*n_FP / n_N), file=out_file)
    
    FP_tm = [int(test_set[test_set["ID"] == FP.index[i]]["TM"]) for i in range(n_FP)]
    N_tm = N[N["TM"]==1]
    print("False positives with TM over negatives with TM: %.2f%%" % (100*sum(FP_tm)/len(N_tm)), file=out_file)
    
    # FALSE NEGATIVES
    
    FN = predictions[(predictions["Prediction"] != predictions["Class"]) & (predictions["Class"] == 1)]
    n_FN = FN.shape[0]
    TP = predictions[(predictions["Prediction"] == predictions["Class"]) & (predictions["Class"] == 1)]
    n_TP = TP.shape[0]
    train_P = train_set[train_set["SP_end"].notna()]
    
    pos_seqs = get_sequences("sets/positives.fasta")
    neg_seqs = get_sequences("sets/negatives.fasta")
    FN_comp = get_aa_composition(pd.DataFrame(FN.index.to_list(), columns=["ID"]), pos_seqs)
    TP_comp = get_aa_composition(pd.DataFrame(TP.index.to_list(), columns=["ID"]), pos_seqs)
    train_P_comp = get_aa_composition(train_P, pos_seqs)
    
    FN_comp = get_total_composition(FN_comp)
    TP_comp = get_total_composition(TP_comp)
    train_P_comp = get_total_composition(train_P_comp)
    #print(FN_comp)
    
    comb = pd.concat([FN_comp,TP_comp,train_P_comp], axis=1)
    comb.columns=["False Negatives","True Positives", "Training positives"]
    plt.figure()
    plt.rcParams.update({'font.size': 12})

    comb.plot(kind="bar", ylabel="Frequency",figsize=(10,8), rot=0)
    plt.savefig("results/analysis/figs/composition.png")
    
    
    
    FN_splen = [int(test_set[test_set["ID"] == FN.index[i]]["SP_end"]) for i in range(n_FN)]
    TP_splen = [int(test_set[test_set["ID"] == TP.index[i]]["SP_end"]) for i in range(n_TP)]
    train_P_splen = train_P["SP_end"]
    
    plt.figure(figsize=(10,8))
    plt.rcParams.update({'font.size': 15})

    #plt.title("SP length distributions")
    sns.kdeplot(data=FN_splen, color="blue", fill=True)
    sns.kdeplot(data=TP_splen, color="orange", fill=True)
    sns.kdeplot(data=train_P_splen, color="red", fill=True)
    plt.legend(["False Negatives", "True Positives", "Training Positives"])
    plt.xlabel("SP length")
    plt.savefig(f"results/analysis/figs/sp_length.png")
    
    
    
    encodings = pd.read_csv("svm_data/encodings_13.tsv", sep="\t", header=0, index_col=0)
    features = encodings.iloc[:,20:-1]
    features_names =['hp_max', 'hp_mean', 'helix_max', 'helix_mean', 'tm_max', 'tm_mean']
    features = features.set_axis(features_names, axis=1)
    FN_features = features.loc[FN.index.to_list()]
    TP_features = features.loc[TP.index.to_list()]
    # make box plots for each feature

    
    fig, axes = plt.subplots(1, 6, figsize=(22, 6))
    plt.rcParams.update({'font.size': 11})

    for i in range(0,len(features_names)):
        f1 = features_names[i]
        sns.boxplot(data=[FN_features[f1], TP_features[f1]], ax=axes[i])
        axes[i].set_xticklabels(["FN", "TP"])
    for i, title in enumerate(["Max Hydrophobicity", "Mean Hydrophobicity","Max Helix Propensity", "Mean Helix Propensity","Max Transmembrane Propensity", "Mean Transmembrane Propensity"]):
        axes[i].set_title(title)
    plt.tight_layout()
    plt.savefig("results/analysis/figs/boxes.png")
    # VH
    print("\n\nVH", file=out_file)
    vh_predictions = pd.read_csv("results/vH_predictions.tsv", sep="\t", header=0, index_col=0)
    
    # FALSE POSITIVES
    
    vh_FP = pd.read_csv("results/vH_FP.tsv", sep="\t", header=0, index_col=0)
    print("False positives over negatives: %.2f%%" % (100*vh_FP.shape[0]/n_N), file=out_file)
    vh_FP_tm = [int(test_set[test_set["ID"] == vh_FP.index[i]]["TM"]) for i in range(vh_FP.shape[0])]
    print("False positives with TM over negatives with TM: %.2f%%" % (100*sum(vh_FP_tm)/len(N_tm)),file=out_file)
    
    # FALSE NEGATIVES
    
    vh_FN = pd.read_csv("results/vH_FN.tsv", sep="\t", header=0, index_col=0)
    vh_TP = vh_predictions[(vh_predictions["Prediction"] == vh_predictions["Class"]) & (vh_predictions["Class"] == 1)]
    with open("results/analysis/FN_logo.txt", "w") as FN_file:
        for FN_id in vh_FN.index:
            pos = int(test_set[test_set["ID"] == FN_id]["SP_end"])
            print(pos_seqs[FN_id][pos-13:pos+2],file=FN_file)
    with open("results/analysis/TP_logo.txt", "w") as TP_file:
        for TP_id in vh_TP.index:
            pos = int(test_set[test_set["ID"] == TP_id]["SP_end"])
            print(pos_seqs[TP_id][pos-13:pos+2],file=TP_file)