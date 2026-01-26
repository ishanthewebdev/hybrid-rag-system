# services/rag_service.py

from loaders.pdf_loader import load_pdf
from chunking.chunker import chunk_text
from retrieval.hybrid import hybrid_rank
from retrieval.rerank import rerank
from context.builder import build_context
from llm.answer import answer_with_context
from llm.validation import validate_answer
from llm.model import get_llm
from llm.query_rewrite import rewrite_query
from context.compressor import compress_context
from embeddings.embedder import embed_texts
from embeddings.embedder import embed_query
from vectorstore.faiss_store import FaissStore
from retrieval.duplicate import duplicate_chunks
import requests

EVAL_API = "http://localhost:8001/evaluate" 

class RAGService:
    def __init__(self):
        # Load once (important for performance)
        # self.store = FaissStore(session_id)
        # # self.text = load_pdf("data/ml2.pdf")
        # self.chunks = chunk_text(self.text)
        self.llm=get_llm()
     # ---------- INGEST ----------
    def ingest(self, pdf_path: str, session_id: str):
        pdf_name = pdf_path.split("/")[-1]
        text = load_pdf(pdf_path)
        chunks = chunk_text(text,pdf_name)

        store = FaissStore(session_id)
        store.add(chunks)

    # def ask(self, query: str,session_id: str) -> str:
    #     assert isinstance(query, str), "Query must be a string"
    #     store = FaissStore(session_id)
    #     rewritten_query = rewrite_query(query, self.llm)
    #     retrieved_chunks = store.search(rewritten_query, top_k=8)
    #     if not retrieved_chunks:
    #         return "I don't know based on the provided document."
    #     # hybrid_results, scores = hybrid_rank(rewritten_query, self.chunks)

    #     # reranked_chunks = rerank(query,self.chunks, hybrid_results)
    #     # reranked_idx = rerank(
    #     #     rewritten_query,
    #     #     # retrieved_chunks,
    #     #     [c["text"] for c in retrieved_chunks],
    #     #     list(range(len(retrieved_chunks)))
    #     # )
    #     reranked_chunks = rerank(rewritten_query, retrieved_chunks, top_k=4)
    #     # top_chunks = [retrieved_chunks[i] for i in reranked_idx]
    #     # context = build_context(top_chunks)
    #     context=build_context([c["text"] for c in reranked_chunks])
    #     # query_embedding = self.embed_model.encode(
    #     #     rewritten_query,
    #     #     convert_to_tensor=True,
    #     #     normalize_embeddings=True
    #     # )

    #     # compressed_context = compress_context(

    #     #     rewritten_query,
    #     #     top_chunks,
    #     #     self.embed_model,
    #     #     query_embedding
    #     #     )

    #     print("=== CONTEXT SENT TO LLM ===")
    #     print(context)
    #     print("=== END CONTEXT ===")
    #     answer = answer_with_context(query, context)
       

    #     verdict = validate_answer(query, context, answer,self.llm)
        

    #     # if verdict.get("relevance",0) <0.5 or not verdict.get("supported",True):
    #     #     return "I don't know based on the provided document."
    #     # if verdict.get("supported", 0) < 0.5:
    #     #     return "I don't know based on the provided document."

    #     return answer
    def ask(self, query: str, session_id: str) -> dict:
     assert isinstance(query, str), "Query must be a string"

    # 1️⃣ Load vector store for this user/session
     store = FaissStore(session_id)

    # 2️⃣ Rewrite query (optional but good)
     rewritten_query = rewrite_query(query, self.llm)

    # 3️⃣ Retrieve from FAISS
     retrieved_chunks = store.search(rewritten_query, top_k=12)
     if not retrieved_chunks:
        return {
            "answer": "I don't know based on the provided document.",
            "context": "",
            "retrieved_chunks": [],
            "reranked_chunks": []
        }
     retrieved_chunks=duplicate_chunks(retrieved_chunks)

    # 4️⃣ Rerank ONLY retrieved chunks
     reranked_chunks = rerank(
        rewritten_query,
        retrieved_chunks,
        top_k=4
    )
     print("=== RETRIEVED CHUNKS ===")
     for c in reranked_chunks:
      print(f"[{c.get('source')} | page {c.get('page')}]")
     print(c["text"][:200])
     print("-----")

    # 5️⃣ Build context (text only)
    #  context = build_context([c["text"] for c in reranked_chunks])
     context=build_context(reranked_chunks)

    # 6️⃣ Generate answer
     answer = answer_with_context(query, context)

    # 7️⃣ (Optional) lightweight validation
     verdict = validate_answer(query, context, answer, self.llm)
     payload = {
            "question": query,
            "answer": answer,
            "context": context
        }

     resp = requests.post(EVAL_API, json=payload,timeout=10)
     if resp.status_code != 200:
       print("❌ Evaluation API failed:", resp.text)
       eval_result = None
     else:
      eval_result = resp.json()

     

        # 3️⃣ log evaluation
     print("\n=== EVALUATION FROM RAG ===")
     print(eval_result)

     return {
            "answer": answer,
            "context": context,
            "evaluation": eval_result
        }
    # 8️⃣ Return everything (THIS IS KEY)
    #  return {
    #     "answer": answer,
    #     "context": context,
    #     "retrieved_chunks": retrieved_chunks,
    #     "reranked_chunks": reranked_chunks,
    #     "verdict": verdict
    # }
     return {
    "query": query,
    "answer": answer,
    "context": context
}

