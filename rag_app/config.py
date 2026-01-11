from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

EMBED_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
RERANK_MODEL = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

LLM = ChatHuggingFace(
    llm=HuggingFaceEndpoint(
        repo_id="HuggingFaceH4/zephyr-7b-beta",
        task="text-generation"
    )
)

MIN_HYBRID_SCORE = 0.2
MAX_CONTEXT_CHARS = 1500

CHUNK_SIZE = 300
CHUNK_OVERLAP = 0

TOP_K = 5
MIN_SCORE = 0.2