import numpy as np
from config import HYBRID_ALPHA

def hybrid_scores(dense, sparse):
    return np.array(dense) + HYBRID_ALPHA * np.array(sparse)
