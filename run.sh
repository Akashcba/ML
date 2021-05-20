export TRAINING_DATA="/Users/my_mac/Documents/Machine Learning/ML/input/train.csv"
export TEST_DATA="/Users/my_mac/Documents/Machine Learning/ML/input/test.csv"
export MODEL_PATH="/Users/my_mac/Documents/Machine Learning/ML/models"


# - binary_classification
# - multiclass_classification
# - multilabel_classification
# - single_col_regression
# - multi_col_regression
# - holdout_[%Value] => Very usefull in time series data(Make shuffle = False )=> and for large datasets.

# Cross Validation
export PROBLEM_TYPE="binary_classification"
export TARGET_COLS="target"
export LABEL_DELIMETER=" "
export NUM_FOLDS="7"

# Categorical Encoding ....
# label
# binary
# ohe
export TYPE="label"
export NA="True"

export MODEL=$1


python -m src.categorical
python -m src.cross_validation

FOLD=0 python -m src.train
FOLD=2 python -m src.train
FOLD=1 python -m src.train
FOLD=3 python -m src.train
FOLD=4 python -m src.train
FOLD=5 python -m src.train
FOLD=6 python -m src.train

python -m src.predict
