from loaders.pdf_loader import load_pdf
from chunking.chunker import chunk_text
from retrieval.bm25 import bm25_scores
from retrieval.dense import dense_scores
from retrieval.hybrid import hybrid_scores
from retrieval.rerank import rerank
from context.context_builder import build_context
from llm.answer import answer_with_rag
from llm.query_rewrite import rewrite_query
from llm.validation import validate_answer
from sentence_transformers import SentenceTransformer
from config import *
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# LLM
llm = HuggingFaceEndpoint(repo_id="HuggingFaceH4/zephyr-7b-beta", task="text-generation")
model = ChatHuggingFace(llm=llm)

text = load_pdf("ml2.pdf")
chunks = chunk_text(text)

query = input("Ask: ")
query = rewrite_query(query, model)

embedder = SentenceTransformer(EMBED_MODEL)
chunk_embs = embedder.encode(chunks, convert_to_tensor=True)
query_emb = embedder.encode(query, convert_to_tensor=True)

dense = dense_scores(query_emb, chunk_embs)
sparse = bm25_scores(chunks, query)
hybrid = hybrid_scores(dense, sparse)

top_idx = sorted(range(len(hybrid)), key=lambda i: hybrid[i], reverse=True)[:TOP_N_RERANK]
reranked = rerank(query, chunks, top_idx)

final_chunks = [chunks[i] for i, _ in reranked[:TOP_K]]
context = build_context(final_chunks)

answer = answer_with_rag(query, context, model)
verdict = validate_answer(query, context, answer, model)

print("\nFINAL ANSWER:\n", answer if verdict["supported"] else "I don't know.")
