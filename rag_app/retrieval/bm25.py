from rank_bm25 import BM25Okapi

# def bm25_scores(chunks, query):
#     tokenized = [c.split() for c in chunks]
#     bm25 = BM25Okapi(tokenized)
#     return bm25.get_scores(query.split())
def bm25_scores(chunks, query):
    if isinstance(query, list):
        query = " ".join(query)

    tokenized_chunks = [c.split() for c in chunks]
    bm25 = BM25Okapi(tokenized_chunks)
    return bm25.get_scores(query.split())
