import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from Bio.SeqUtils.ProtParam import ProteinAnalysis

def make_length_plot(df_pos: pd.DataFrame, df_neg: pd.DataFrame, fig_name, title):
    plt.figure(figsize=(10,8))
    #plt.title(title)
    plt.rcParams.update({'font.size': 16})
    sns.kdeplot(data=df_pos, x="Length", color="blue", fill=True)
    sns.kdeplot(data=df_neg, x="Length", color="orange", fill=True)
    plt.xlim(right=6000, left=-500)
    plt.legend(["Positives", "Negatives"])
    plt.savefig(f"figs/{fig_name}.png")


def make_sp_length_plot(df, fig_name, title):
    plt.figure(figsize=(10,10))
    sns.displot(df, x="SP_end", stat="density", kde=True)
    plt.title(title, loc="center", pad=20, x=0.5)
    plt.xlim(right=70)
    plt.tight_layout() 
    plt.savefig(f"figs/{fig_name}.png")
    
def make_sp_length_plot_2(df1, df2, fig_name, title):
    plt.figure(figsize=(8,5))
    plt.rcParams.update({'font.size': 11})
    #plt.title(title, loc="center", pad=20, x=0.5)
    plt.xlabel("Signal peptide length")
    sns.kdeplot(data=df1, x="SP_end", color='blue', fill=True)
    sns.kdeplot(data=df2, x="SP_end", color='orange', fill=True)
    plt.tight_layout() 
    plt.legend(["Training set", "Benchmarking set"])
    plt.savefig(f"figs/{fig_name}.png")
    
def make_tax_hist(train, test):
    organism_counts_train = train['Organism'].value_counts()[:7]
    organism_counts_train["Other"] = train['Organism'].value_counts()[7:].sum()
    total_org_train = sum(organism_counts_train)
    organism_counts_train *= 100 / sum(organism_counts_train) 
    organism_counts_test = test['Organism'].value_counts()[:7]
    organism_counts_test["Other"] = test['Organism'].value_counts()[7:].sum()
    total_org_test = sum(organism_counts_test)
    organism_counts_test *= 100 / sum(organism_counts_test) 


    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    plt.rcParams.update({'font.size': 13})

    # Plot histogram for train data
    labels_train = [f"{x.split()[0][0]}. {x.split()[1]}" for x in organism_counts_train.index[:-1]]
    labels_train.append("Other")
    labels_test = [f"{x.split()[0][0]}. {x.split()[1]}" for x in organism_counts_test.index[:-1]]
    labels_test.append("Other")
    
    ax1.bar(organism_counts_train.index, organism_counts_train, color='skyblue')
    ax1.set_title("Training set species composition")
    ax1.set_xticklabels(labels_train, rotation=45, ha='right')
    ax2.set_xticklabels(labels_test, rotation=45, ha='right')
    ax1.set_ylabel("Percentage (%)")
    ax1.set_xlabel("Species")
    ax1.set_ylim(0, 35)

    # Add value labels to bars for train data
    for i, value in enumerate(organism_counts_train):
        cnt = int(value / 100 * total_org_train)
        ax1.text(i, value + 1, f'{value:.1f}%\n({cnt})', ha='center')

    # Plot histogram for test data
    ax2.bar(organism_counts_test.index, organism_counts_test, color='lightgreen')
    ax2.set_title("Benchmarking set species composition")
    #ax2.set_ylabel("Counts")
    ax2.set_xlabel("Species")
    ax2.set_ylim(0, 35)

    # Add value labels to bars for test data
    for i, value in enumerate(organism_counts_test):
        cnt = int(value / 100 * total_org_test)
        ax2.text(i, value + 1, f'{value:.1f}%\n({cnt})', ha='center')

    # Adjust layout
    plt.tight_layout()
    plt.savefig(f"figs/organism_pie_charts.png", bbox_inches='tight')
    
    
    
    
    
    kingdom_counts_train = train['Kingdom'].value_counts()
    total_king_train = sum(kingdom_counts_train)
    kingdom_counts_train *= 100/total_king_train
    kingdom_counts_test = test['Kingdom'].value_counts()
    total_king_test = sum(kingdom_counts_test)
    kingdom_counts_test *= 100/total_king_test

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    plt.rcParams.update({'font.size': 11})

    # Plot histogram for train data
    labels_train = kingdom_counts_train.index
    labels_test = kingdom_counts_test.index
    
    ax1.bar(kingdom_counts_train.index, kingdom_counts_train, color='skyblue')
    ax1.set_title("Training set kingdom composition")
    #ax1.set_xticklabels(labels_train, rotation=45, ha='right')
    #ax2.set_xticklabels(labels_test, rotation=45, ha='right')
    ax1.set_ylabel("Percentage (%)")
    ax1.set_xlabel("Kingdom")
    ax1.set_ylim(0, 60)

    # Add value labels to bars for train data
    for i, value in enumerate(kingdom_counts_train):
        cnt = int(value / 100 * total_org_train)
        ax1.text(i, value + 1, f'{value:.1f}%\n({cnt})', ha='center')

    # Plot histogram for test data
    ax2.bar(kingdom_counts_test.index, kingdom_counts_test, color='lightgreen')
    ax2.set_title("Benchmarking set kingdom composition")
    #ax2.set_ylabel("Counts")
    ax2.set_xlabel("Kingdom")
    ax2.set_ylim(0, 60)

    # Add value labels to bars for test data
    for i, value in enumerate(kingdom_counts_test):
        cnt = int(value / 100 * total_org_test)
        ax2.text(i, value + 1, f'{value:.1f}%\n({cnt})', ha='center')

    # Adjust layout
    plt.tight_layout()
    plt.savefig(f"figs/kingdom_pie_charts.png", bbox_inches='tight')
    
    


    
    
def get_sp_composition(df: pd.DataFrame, fasta_name):
    seqs = {}
    with open(fasta_name) as fasta:
        for line in fasta:
            if line[0]==">":
                id = line[1:].rstrip()
            else:
                seqs[id] = line.rstrip()
    
    sp_seqs = {}
    for _, (id, end) in df.loc[:,["ID","SP_end"]].iterrows():
        X = ProteinAnalysis(seqs[id][:end])
        sp_seqs[id]=X.get_amino_acids_percent()

    return pd.DataFrame.from_dict(sp_seqs,orient='index')

def make_composition_plot(df1: pd.DataFrame, df2: pd.DataFrame, fig_name):
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
    df1_composition = get_sp_composition(df1, "sets/positives.fasta")
    df1_mean_composition = df1_composition.mean()
    df2_composition = get_sp_composition(df2, "sets/positives.fasta")
    df2_mean_composition = df2_composition.mean()
    
    comb = pd.concat([pd.DataFrame(swissprot_comp),pd.DataFrame(df1_mean_composition),pd.DataFrame(df2_mean_composition)], axis=1)
    comb.columns=["Swiss-Prot","Training set", "Benchmarking set"]
    comb = comb.reindex(["G","A","V","P","L","I","M","F","W","Y","S","T","C","N","Q","H","D","E","K","R"])
    plt.rcParams.update({'font.size': 13})
    comb.plot(kind="bar", ylabel="Frequency",figsize=(10,8), rot=0)
    plt.savefig(f"figs/{fig_name}.png")
    
def make_logo(df: pd.DataFrame, out_file_name):
    out_file = open(out_file_name, "w")
    with open("positives.fasta") as fasta:
        found = False
        for line in fasta:
            if line[0]==">":
                id = line[1:].rstrip()
                if (df["ID"]==id).any():
                    cleavage = int(df[df["ID"]==id]["SP_end"])
                    found = True
            elif found:
                found = False
                print(line[cleavage-13:cleavage+2], file=out_file)
    out_file.close()
    
if __name__ == "__main__":
    train_positives = pd.read_csv("sets/train_positives.tsv", sep="\t", index_col=0)
    train_negatives = pd.read_csv("sets/train_negatives.tsv", sep="\t", index_col=0)
    test_positives = pd.read_csv("sets/test_positives.tsv", sep="\t", index_col=0)
    test_negatives = pd.read_csv("sets/test_negatives.tsv", sep="\t", index_col=0)
    train = pd.concat([train_positives, train_negatives], ignore_index=True)
    test = pd.concat([test_positives, test_negatives], ignore_index=True)
    
    sns.set_style("whitegrid")
    sns.color_palette("bright")
    
    # 1. Protein length (pos vs neg) OK
    make_length_plot(train_positives, train_negatives, "train_length", "Training set sequence length distribution")
    make_length_plot(test_positives, test_negatives, "test_length", "Benchmarking set sequence length distribution")

    # 2. SP lengths OK
    make_sp_length_plot_2(train_positives, test_positives, "sp_length", "Signal peptide length distribution")
    
    # 3. A.A. composition OK
    make_composition_plot(train_positives, test_positives, "aa_composition")
    
    # 4. Taxonomic classification OK
    make_tax_hist(train, test)

    # 5. Sequence logo
    make_logo(train_positives,"logos/train_logo.txt")
    make_logo(test_positives,"logos/test_logo.txt")
    




