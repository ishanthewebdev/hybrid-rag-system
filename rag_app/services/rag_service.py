# services/rag_service.py

from loaders.pdf_loader import load_pdf
from chunking.chunker import chunk_text
from retrieval.hybrid import hybrid_rank
from retrieval.rerank import rerank
from context.builder import build_context
from llm.answer import answer_with_context
from llm.validation import validate_answer
from llm.model import get_llm



class RAGService:
    def __init__(self):
        # Load once (important for performance)
        self.text = load_pdf("data/ml2.pdf")
        self.chunks = chunk_text(self.text)
        self.llm=get_llm()

    def ask(self, query: str) -> str:
        assert isinstance(query, str), "Query must be a string"
        hybrid_results, scores = hybrid_rank(query, self.chunks)

        reranked_chunks = rerank(query,self.chunks, hybrid_results)
        top_chunks = [self.chunks[i] for i in reranked_chunks]
        context = build_context(top_chunks)

        answer = answer_with_context(query, context)
        print("=== CONTEXT SENT TO LLM ===")
        print(context)
        print("=== END CONTEXT ===")

        verdict = validate_answer(query, context, answer,self.llm)
        

        if not verdict["relevance"]<0.5:
            return "I don't know based on the provided document."

        return answer
