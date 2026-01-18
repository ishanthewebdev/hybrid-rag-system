import faiss
import pickle
import os
from sentence_transformers import SentenceTransformer
from config import EMBED_MODEL_NAME

def build_index(chunks, session_id):
    model = SentenceTransformer(EMBED_MODEL_NAME)
    embeddings = model.encode(chunks)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    base = f"temp_indexes/{session_id}"
    os.makedirs(base, exist_ok=True)

    faiss.write_index(index, f"{base}/faiss.index")

    with open(f"{base}/chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
