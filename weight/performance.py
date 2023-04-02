from sklearn.metrics import roc_auc_score
import numpy as np

def calculate_roc_auc(model_pipe, X, y):
    """Calculate roc auc score. 
    
    Parameters:
    ===========
    model_pipe: sklearn model or pipeline
    X: features
    y: true target
    """
    predict_prob=model_pipe.predict(X)

    predict_classes=np.argmax(predict_prob,axis=1)
    return roc_auc_score(y, predict_classes)