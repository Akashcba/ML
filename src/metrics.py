
'''
Binary Classification Metrics
-> Accuracy
-> Precision
-> Recall
-> F1-Score
-> AUC (Area Under ROC curve), Receiver Operating Characteristics
-> logloss
'''
# Theory behind Loss Calculation (Fo binary case)
'''
target = [0,1]; 0 = Negative, 1 = Positive

True Positive (TP) = Target is + and we predicted +
True Negative (TN) = Target is - and we predicted -
False Positive (FP) = Target is - but we predicted +
False Negative (FN) = Target is + but we predicted -

Accuracy = Correct_Predictions / Total_No_Of_Samples
         = (TP + TN)/(TP+FP+FN+TN)
Precision = How precise the Model is = Out of all Positives PREDICTED how many were truly Positive.
          = TP/(TP + FP) = Precision High => False Positive Rate is Low -> That is what we want (Predict less False +)
Recall = TP/(TP+FN) = Out of all REAL + Values how many we predicted Correctly.
F1 = (weighted Average of Precision and Recall)
   = 2*Precision*Recall/(Precision+Recall)
   = 2.TP/(2.TP + FP + FN)

TPR = True Positive Rate
    = TP/(TP + FN) = Recall
FPR = False Positive Rate = Out of all Negative samples how many were incorrectly predicted as +
    = FP/(TN + FP)

'''
# [0,   0,    1,   0,   1,   1]   => True Labels
# [0.6, 0.4, 0.7, 0.3, 0.5, 0.9]  => Predicted Labels
'''
AUC is the Graph b/w TPR and FPR for different values of THRESHOLD
The area under this curve is the AUC
AUC = 1 -> great model
AUC = 0.5 -> Random Model
Given any + sample from the dataset and any - sample in the dataset
what is the prob that the + sample will rank higher than - sample
this values is the AUC
'''
# LogLoss
'''
logloss = -(y.log(P) + (1-y).log(1-P)); P = prediction
Penalizes quite high for wrong prediction.
For all samples is just average of individual sample loglosses
'''

