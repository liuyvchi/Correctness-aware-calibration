import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import label_binarize


def brier_score(y_pred, y):
    # y_one_hot = label_binarize(y, classes=np.arange(len(y_pred[0])))
    loss = mean_squared_error(y, y_pred)
    return loss

def top1_brier_score(score, acc):
    loss = np.mean((score - acc)**2)
    return loss