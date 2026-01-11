from sentence_transformers import util
from embeddings.embedder import embed_texts, embed_query

def dense_scores(chunks, query):
    chunk_emb = embed_texts(chunks)
    query_emb = embed_query(query)
    return util.cos_sim(query_emb, chunk_emb)[0].cpu().tolist()
