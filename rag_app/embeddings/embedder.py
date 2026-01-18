from config import EMBED_MODEL

def embed_texts(texts):
    return EMBED_MODEL.encode(texts, convert_to_tensor=True,normalize_embeddings=True)

def embed_query(query):
    return EMBED_MODEL.encode(query, convert_to_tensor=True,normalize_embeddings=True)
