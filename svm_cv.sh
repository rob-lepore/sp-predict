printf "Feature set\tC\tgamma\tMCC\tF1\tPrecision\tRecall\tAccuracy\n" > svm_data/cv_summary.tsv
for f_set in {1..16}; do
    echo ""
    echo "Feature set ${f_set}"
    python3 "svm_cross_validate.py" "svm_data/encodings_${f_set}.tsv" "svm_data/models/svm_${f_set}.pkl.gz"
    echo "***********************************"
done