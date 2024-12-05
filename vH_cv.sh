rm folds/*.tsv
rm vH_data/*.tsv
rm vH_data/models/*.tsv

echo "Making folds"
python3 make_folds.py sets/train.tsv

for n in {1..5}; do
    echo ""
    echo "Run ${n}: training"
    python3 vH-train.py "folds/fold_${n}_train.tsv" "vH_data/models/vH_model_${n}.tsv"
    echo "Run ${n}: scoring validation set"
    python3 vH-predict.py "vH_data/models/vH_model_${n}.tsv" "folds/fold_${n}_validation.tsv" <(cat sets/positives.fasta sets/negatives.fasta) "vH_data/predictions_${n}_validation.tsv"
    echo "Run ${n}: scoring test set"
    python3 vH-predict.py "vH_data/models/vH_model_${n}.tsv" "folds/fold_${n}_test.tsv" <(cat sets/positives.fasta sets/negatives.fasta) "vH_data/predictions_${n}_test.tsv"
done

echo ""
echo "Evaluating performance: "
python3 performance.py

echo ""
echo "Training on the complete training set"
python3 vH-train.py sets/train_positives.tsv vH_data/models/vH_model_train.tsv