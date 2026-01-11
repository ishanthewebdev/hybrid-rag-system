from loaders.pdf_loader import load_pdf
from chunking.chunker import chunk_text
from retrieval.hybrid import hybrid_rank
from context.builder import build_context
from llm.answer import answer_with_context
from utils.guards import is_context_valid
from config import MAX_CONTEXT_CHARS
from llm.model import get_llm   # if you created this helper
llm = get_llm() 
PDF_PATH = "data/ml2.pdf"

def run_rag(query):
    text = load_pdf(PDF_PATH)
    chunks = chunk_text(text)

    ranking, scores = hybrid_rank(chunks, query)
    top_chunks = [chunks[i] for i in ranking[:5]]

    context = build_context(top_chunks, MAX_CONTEXT_CHARS)

    if not is_context_valid(context):
        return "I don't know based on the provided document."

    return answer_with_context(query, context)

if __name__ == "__main__":
    q = input("Ask question: ")
    print(run_rag(q))
