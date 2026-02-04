import numpy as np
import pandas as pd

from config import *
from models import train_catboost_ensemble
from objectives import (
    predict_objective1_ensemble,
    compute_objective2_cost
)



def is_non_dominated(points: np.ndarray) -> np.ndarray:
    """
    Non-dominated mask for minimization problems.
    """
    n = points.shape[0]
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        if not mask[i]:
            continue
        dominated = np.all(points <= points[i], axis=1) & \
                    np.any(points < points[i], axis=1)
        dominated[i] = False
        mask[dominated] = False
    return mask


def hypervolume_2d(front: np.ndarray, ref: np.ndarray) -> float:
   
    if front.size == 0:
        return 0.0

    idx = np.argsort(front[:, 0])
    front = front[idx]

    hv = 0.0
    prev_y = ref[1]
    for x, y in front:
        width = max(ref[0] - x, 0.0)
        height = max(prev_y - y, 0.0)
        hv += width * height
        prev_y = min(prev_y, y)

    return hv




def select_candidates_by_ehvi(
    pred_obj1_pool,    
    pred_obj1_train,   
    obj2_pool,          # [n_pool]
    obj2_train,         
    topN: int,
    q: int
):
    n_models, n_pool = pred_obj1_pool.shape

    
    pareto_fronts = []
    hv_current = np.zeros(n_models)

    for k in range(n_models):
        train_points = np.column_stack([
            pred_obj1_train[k],
            obj2_train
        ])
        front = train_points[is_non_dominated(train_points)]
        pareto_fronts.append(front)

  
    all_fronts = np.vstack(pareto_fronts)
    ref = all_fronts.max(axis=0) * 1.05

    for k in range(n_models):
        hv_current[k] = hypervolume_2d(pareto_fronts[k], ref)

  
    mean_obj1 = pred_obj1_pool.mean(axis=0)
    scalar_score = mean_obj1 + obj2_pool
    coarse_idx = np.argsort(scalar_score)[:topN]

    ehvi = np.zeros(len(coarse_idx))

    for i, idx in enumerate(coarse_idx):
        for k in range(n_models):
            cand_point = np.array([[pred_obj1_pool[k, idx], obj2_pool[idx]]])
            combined = np.vstack([pareto_fronts[k], cand_point])
            new_front = combined[is_non_dominated(combined)]

            hv_new = hypervolume_2d(new_front, ref)
            ehvi[i] += max(hv_new - hv_current[k], 0.0)

        ehvi[i] /= n_models

    # ---- select top-q ----
    best_rel = np.argsort(-ehvi)[:q]
    best_idx = coarse_idx[best_rel]

    return best_idx, ehvi[best_rel]


# ============================================================
# Main optimization loop
# ============================================================

def main():
    print("Loading data ...")

    pool_df = pd.read_excel(POOL_PATH)
    X_pool = pool_df.iloc[:, 1:20].values

    train_df = pd.read_excel(TRAIN_PATH)
    X_train = train_df.iloc[:, 2:20].values
    y_train_obj1 = train_df.iloc[:, 20].values

    print("Training CatBoost ensemble ...")

    models = train_catboost_ensemble(
        X_train,
        y_train_obj1,
        n_models=N_MODELS,
        iterations=CB_ITERATIONS,
        bootstrap=USE_BOOTSTRAP,
        random_seed=RANDOM_SEED
    )

    # ---- objectives ----
    pred_obj1_pool = predict_objective1_ensemble(models, X_pool)
    pred_obj1_train = predict_objective1_ensemble(models, X_train)

    obj2_pool = compute_objective2_cost(X_pool, ELEMENT_PRICES)
    obj2_train = compute_objective2_cost(X_train, ELEMENT_PRICES)

    # ---- EHVI selection ----
    chosen_idx, scores = select_candidates_by_ehvi(
        pred_obj1_pool,
        pred_obj1_train,
        obj2_pool,
        obj2_train,
        TOPN_COARSE,
        Q_SELECT
    )

    print("Selected pool indices:", chosen_idx)

    np.savetxt(
        OUTPUT_CSV,
        X_pool[chosen_idx],
        delimiter=",",
        fmt="%.8f"
    )

    print("Saved results to:", OUTPUT_CSV)


if __name__ == "__main__":
    main()
