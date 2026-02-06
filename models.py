"""

This module defines the surrogate model ensemble used for
predicting the stochastic objective (η).

"""
import numpy as np
from catboost import CatBoostRegressor
from typing import List

def train_catboost_ensemble(
    X: np.ndarray,
    y: np.ndarray,
    n_models: int,
    iterations: int,
    bootstrap: bool,
    random_seed: int
) -> List[CatBoostRegressor]:

    n = X.shape[0]
    models = []

    for m in range(n_models):
        if bootstrap:
            idx = np.random.choice(n, n, replace=True)
            Xm, ym = X[idx], y[idx]
        else:
            Xm, ym = X, y

        model = CatBoostRegressor(
            iterations=iterations,
            depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bylevel=0.8,
            random_seed=random_seed + m,
            verbose=False
        )
        model.fit(Xm, ym)
        models.append(model)

    return models
