import numpy as np

def predict_objective1_ensemble(models, X):
   
    preds = np.stack([m.predict(X) for m in models], axis=0)
    return preds


def compute_objective2_cost(X, prices):
   
    return X @ prices
