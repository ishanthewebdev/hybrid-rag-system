# import numpy as np
# from config import HYBRID_ALPHA

# def hybrid_scores(dense, sparse):
#     return np.array(dense) + HYBRID_ALPHA * np.array(sparse)
import numpy as np
from retrieval.bm25 import bm25_scores
from retrieval.dense import dense_scores

def hybrid_rank(chunks, query):
    if isinstance(query, list):
        query = " ".join(query)
    
    bm25 = bm25_scores(chunks, query)
    dense = dense_scores(chunks, query)

    scores = np.array(dense) + 0.5 * np.array(bm25)
    ranking = np.argsort(-scores)
    return ranking, scores
