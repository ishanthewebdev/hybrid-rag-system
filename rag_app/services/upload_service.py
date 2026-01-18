from loaders.pdf_loader import load_pdf
from chunking.chunker import chunk_text
from vectorstore.index_manager import build_index

def process_pdf(pdf_path, session_id):
    text = load_pdf(pdf_path)
    chunks = chunk_text(text)
    build_index(chunks, session_id)
