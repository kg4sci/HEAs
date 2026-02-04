import numpy as np

# ---------------- Paths ----------------
POOL_PATH = ''
TRAIN_PATH = ''
OUTPUT_CSV = ''

# ---------------- AL params ----------------
N_MODELS = 20
CB_ITERATIONS = 500
TOPN_COARSE = 200
Q_SELECT = 5
RANDOM_SEED = 0
USE_BOOTSTRAP = True
KAPPA_LCB = 1.96

# ---------------- Objective setting ----------------
OUTCOME_DIM = 2 

ELEMENT_PRICES = np.array([
    # Cr, Mn, Fe, Co, Ni, Cu, Zn, Mo, Ru,
    # Rh, Pd, Ag, W, Re, Os, Ir, Pt, Au
    9.4,
1.82,
0.424,
32.8,
13.9,
6,
2.55,
40.1,
10500,
147000,
49500,
521,
35.3,
3580,
12000,
55850,
27800,
75430
])