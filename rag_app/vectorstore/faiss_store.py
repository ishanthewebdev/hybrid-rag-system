# # vectorstore/faiss_store.py

# import faiss
# import pickle
# import os
# from config import EMBED_MODEL

# class FaissStore:
#     def __init__(self, session_id: str):
#         self.session_id = session_id
#         self.index_path = f"temp_indexes/{session_id}"
#         self.index_file = f"{self.index_path}/faiss.index"
#         self.chunks_file = f"{self.index_path}/chunks.pkl"

#     def build(self, chunks: list[str]):
#         os.makedirs(self.index_path, exist_ok=True)

#         embeddings = EMBED_MODEL.encode(chunks, convert_to_numpy=True)
#         dim = embeddings.shape[1]

#         index = faiss.IndexFlatL2(dim)
#         index.add(embeddings)

#         faiss.write_index(index, self.index_file)

#         with open(self.chunks_file, "wb") as f:
#             pickle.dump(chunks, f)

#     def load(self):
#         index = faiss.read_index(self.index_file)
#         with open(self.chunks_file, "rb") as f:
#             chunks = pickle.load(f)
#         return index, chunks

#     def search(self, query: str, top_k: int = 5):
#         index, chunks = self.load()
#         query_vec = EMBED_MODEL.encode([query], convert_to_numpy=True)
#         scores, ids = index.search(query_vec, top_k)
#         return [chunks[i] for i in ids[0]]

import faiss
import numpy as np
import pickle
from embeddings.embedder import embed_texts,embed_query
import os

class FaissStore:
    def __init__(self, session_id: str, dim: int = 384):
        self.index_path = f"storage/{session_id}.index"
        self.meta_path = f"storage/{session_id}.pkl"
        self.dim = dim

        if self._exists():
            self._load()
        else:
            self.index = faiss.IndexFlatL2(dim)
            self.metadata = []

    def _exists(self):
        # return False
        return os.path.exists(self.index_path) and os.path.exists(self.meta_path)
    def _load(self):
        self.index = faiss.read_index(self.index_path)
        with open(self.meta_path, "rb") as f:
            self.metadata = pickle.load(f)

    def _save(self):
        os.makedirs("storage", exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "wb") as f:
            pickle.dump(self.metadata, f)

    def add(self, chunks: list[dict]):
        texts = [c["text"] for c in chunks]
        embeddings = embed_texts(texts)
        embeddings = np.array(embeddings).astype("float32")

        self.index.add(embeddings)
        self.metadata.extend(chunks)

        # 🔒 ALWAYS save after add
        self._save()
        # embeddings = embed_texts(texts)

        # self.index.add(np.array(embeddings).astype("float32"))
        # self.metadata.extend(chunks)

    def search(self, query: str, top_k: int = 8):
        if self.index.ntotal == 0:
            return []
        q_emb = embed_query(query)
        if hasattr(q_emb, "cpu"):
         q_emb = q_emb.cpu().numpy()
        if len(q_emb.shape) == 1:
         q_emb = q_emb.reshape(1, -1)
        q_emb = q_emb.astype("float32")
        D, I = self.index.search(q_emb, top_k)

        results = []
        for i in I[0]:
            if 0 <= i < len(self.metadata):
                results.append(self.metadata[i])

        return results
