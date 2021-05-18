export TRAINING_DATA=input/train_folds.csv
export TEST_DATA=input/test.csv

export EPOCHS=5
export BATCH_SIZE=4
export LR=0.00002

export MODEL=$1

# FOLD=0 python -m src.train
# FOLD=1 python -m src.train
# FOLD=2 python -m src.train
# FOLD=3 python -m src.train
# FOLD=4 python -m src.train
python -m src.predict