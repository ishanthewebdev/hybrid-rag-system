from sentence_transformers import util

def dense_scores(query_emb, chunk_embs):
    return util.cos_sim(query_emb, chunk_embs)[0].cpu().tolist()
