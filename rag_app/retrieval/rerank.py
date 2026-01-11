# from sentence_transformers import CrossEncoder
# from config import RERANK_MODEL

# reranker = CrossEncoder(RERANK_MODEL)

# def rerank(query, chunks, indices):
#     pairs = [(query, chunks[i]) for i in indices]
#     scores = reranker.predict(pairs)
#     ranked = sorted(zip(indices, scores), key=lambda x: x[1], reverse=True)
#     return ranked

from config import RERANK_MODEL
from config import TOP_K
import numpy as np


def rerank(query, chunks, candidate_idx,TOP_k=5):
    pairs = [(query, chunks[i]) for i in candidate_idx]
    scores = RERANK_MODEL.predict(pairs)
    order = np.argsort(-scores)
    final_idx = [candidate_idx[i] for i in order[:TOP_K]]
    return final_idx
