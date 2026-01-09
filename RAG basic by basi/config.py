# config.py

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

CHUNK_SIZE = 300
CHUNK_OVERLAP = 0

TOP_K = 5
TOP_N_RERANK = 10

HYBRID_ALPHA = 0.5
MIN_HYBRID_SCORE = 0.2

MAX_CONTEXT_CHARS = 1500
MIN_RELEVANCE = 0.6
