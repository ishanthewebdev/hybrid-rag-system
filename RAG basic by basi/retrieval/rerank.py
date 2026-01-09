from sentence_transformers import CrossEncoder
from config import RERANK_MODEL

reranker = CrossEncoder(RERANK_MODEL)

def rerank(query, chunks, indices):
    pairs = [(query, chunks[i]) for i in indices]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(indices, scores), key=lambda x: x[1], reverse=True)
    return ranked
