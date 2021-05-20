import os
import pandas as pd
from sklearn import ensemble
from sklearn import preprocessing
from sklearn import metrics
import joblib
import numpy as np
import time

from . import dispatcher


TEST_DATA = os.environ.get("TEST_DATA")
MODEL = os.environ.get("MODEL")
PATH = os.environ.get("MODEL_PATH")
NUM_FOLDS = int(os.environ.get("NUM_FOLDS"))

def predict(test_data_path, model_type, model_path=PATH):
    df = pd.read_csv(test_data_path)
    test_idx = df["id"].values
    predictions = None

    for FOLD in range(NUM_FOLDS):
        df = pd.read_csv(test_data_path)
        #encoders = joblib.load(os.path.join(model_path, f"{model_type}_{FOLD}_label_encoder.pkl"))
        #cols = joblib.load(os.path.join(model_path, f"{model_type}_{FOLD}_columns.pkl"))
        cols = [c for c in df.columns if c not in ["id", "target","kfold"]]
        '''
        for c in encoders:
            lbl = encoders[c]
            df.loc[:, c] = df.loc[:, c].astype(str).fillna("NONE")
            df.loc[:, c] = lbl.transform(df[c].values.tolist())
        '''
        clf = joblib.load(os.path.join(model_path, f"{model_type}_{FOLD}.pkl"))
        
        df = df[cols]
        preds = clf.predict_proba(df)[:, 1]

        if FOLD == 0:
            predictions = preds
        else:
            predictions += preds
    
    predictions /= NUM_FOLDS

    sub = pd.DataFrame(np.column_stack((test_idx, predictions)), columns=["id", "target"])
    return sub    



if __name__ == "__main__":
    print("\nPridicting The Values ......")
    time.sleep(7)
    submission = predict(test_data_path=TEST_DATA, 
                         model_type=MODEL)
    submission.loc[:, "id"] = submission.loc[:, "id"].astype(int)
    submission.to_csv(f"/Users/my_mac/Documents/Machine Learning/ML/input/{MODEL}_submission.csv", index=False)
