from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP
print("chunker loaded")
print(dir())


def chunk_text(text:str,pdf_name:str)->list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = []
    for i, chunk in enumerate(splitter.split_text(text)):
        chunks.append({
            "text": chunk,
            "pdf_name": pdf_name,
            "chunk_id": i
        })
    return chunks
