import os
import pandas as pd
from sklearn import preprocessing

from sklearn import ensemble
from sklearn import metrics

from . import dispatcher
import joblib

TRAINING_DATA = os.environ.get("TRAINING_DATA")
FOLD = int(os.environ.get("FOLD"))
MODEL = os.environ.get("MODEL")

Fold_Mapping = {
    0 : [1,2,3,4],
    1 : [0,2,3,4],
    2 : [0,1,3,4],
    3 : [0,1,2,4],
    4 : [0,1,2,3]
}

if __name__ == "__main__":
    df = pd.read_csv(TRAINING_DATA)
    train_df = df[df.kfold.isin(Fold_Mapping.get(FOLD))]
    valid_df = df[df.kfold==FOLD]

    ytrain = train_df.target.values
    yvalid = valid_df.target.values

    train_df = train_df.drop(["id", "target", "kfold"], axis=1)
    valid_df = valid_df.drop(["id", "target", "kfold"], axis=1)

## Add Same column sequence in both
    valid_df = valid_df[train_df.columns]

    label_encoders =[]
    for c in train_df.columns:
        lbl = preprocessing.LabelEncoder()
        lbl.fit(train_df[c].values.tolist() + valid_df[c].values.tolist())
        train_df.loc[:,c] = lbl.transform(train_df[c].values.tolist())
        valid_df.loc[:,c] = lbl.transform(valid_df[c].values.tolist())
        label_encoders.append((c, lbl))

    # Data is ready to train
    clf = dispatcher.MODELS[MODEL]
    clf.fit(train_df, ytrain)
    preds = clf.predict_proba(valid_df)[:, 1]
    #print(preds)
    print(metrics.roc_auc_score(yvalid, preds))

    ## Storing the data for predict.py
    joblib.dump(label_encoders, f"/Users/my_mac/Documents/Machine Learning/ML/models/{MODEL}_{FOLD}_label_encoder.pkl")
    joblib.dump(clf, f"/Users/my_mac/Documents/Machine Learning/ML/models/{MODEL}_{FOLD}.pkl")
    joblib.dump(train_df.columns, f"/Users/my_mac/Documents/Machine Learning/ML/models/{MODEL}_{FOLD}_columns.pkl")

