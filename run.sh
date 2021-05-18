export TRAINING_DATA="/Users/my_mac/Documents/Machine Learning/ML/input/train_folds.csv"
export TEST_DATA="/Users/my_mac/Documents/Machine Learning/ML/input/test.csv"
export MODEL_PATH="/Users/my_mac/Documents/Machine Learning/ML/models"

export MODEL=$1

FOLD=0 python -m src.train
FOLD=1 python -m src.train
FOLD=2 python -m src.train
FOLD=3 python -m src.train
FOLD=4 python -m src.train

python -m src.predict