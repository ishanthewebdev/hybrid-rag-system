from sentence_transformers import util, SentenceTransformer,CrossEncoder
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from rank_bm25 import BM250kapi
from rank_bm25 import BM25Okapi
import pdfplumber
import pytesseract
from PIL import ImageChops
import re
from huggingface_hub import InferenceClient
import json
import numpy as np
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def is_scanned(page):
    text=page.extract_text() or''
    return len(text.strip())<50

def ocr_page(page):
    img = page.to_image(resolution=300).original
    text=pytesseract.image_to_string(img,lang='eng')
    return text

def clean_text(text):
    text=re.sub(r"-\n","",text)
    text=re.sub(r"\n{3,}","\n\n",text)
    return text.strip()

def load_pdf(path):
    final_pages=[]

    with pdfplumber.open(path) as pdf:
        for i,page in enumerate(pdf.pages):
            if is_scanned(page):
                print(f"page {i}:OCR Mode")
                raw=ocr_page(page)
            else:
                print(f"Page{i}:Normal Text")
                raw=page.extract_text() or ""

            cleaned=clean_text(raw)
            final_pages.append(cleaned)

        return "\n\n=== PAGE BREAK ===\n\n".join(final_pages)

text= load_pdf("small_rag_test.pdf")
print("FINAL RESULT LENGTH:", len(text))
print(text[:2000])

# cleaned text  ho chuka hai now its time to do chunking

splitter=RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=0,
)

chunks=splitter.split_text(text)
query=input("ask your query about the pdf")
# chunk making ho chuki hai now its time to embedding and retrival 
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
emb_chunks= model.encode(chunks,convert_to_tensor=True)
emb_query= model.encode(query,convert_to_tensor=True)
#cosine similarity
# dense_scores = util.cos_sim(emb_query, emb_chunks).gpu().tolist()[0]
#here we are using gpu to store the list, as cosin sim returns a list are we are seeing its 0th elemnet

#now its time to rerank by hybrid reranking method
#here we are splitting chunks into list of strings as bm25 take it like that

tokenized_chunks=[c.split() for c in chunks]
# print(tokenized_chunks)

def re_ranking(chunks,query,emb_query,emb_chunks):
    tokenized_chunks=[c.split() for c in chunks]
    
    #bm25 is for word to word find
    bm25=BM25Okapi(tokenized_chunks)
    bm25_scores=bm25.get_scores(query.split())
    dense_scores = util.cos_sim(emb_query, emb_chunks).cpu().tolist()[0]
    hybrid_scores = np.array(dense_scores) + 0.5 * np.array(bm25_scores)
    ranking = np.argsort(-hybrid_scores)
    return ranking , hybrid_scores

print("\n=== HYBRID RESULTS ===")
# ranking=re_ranking(chunks,query,emb_query,emb_query)
ranking, hybrid_scores = re_ranking(chunks, query, emb_query, emb_chunks)

# print(ranking)
for idx in ranking[:5]:   # top-5
    print(f"Score: {hybrid_scores[idx]:.4f}")
    print("DOC:", chunks[idx][:300].replace("\n", " "), "...")

#now lets do cross encoding for more pefect query context retrival

def cross_encoding(ranking,query,chunks):
    top_n=10
    candidate_idx= ranking[:top_n]
    reranker=CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    pairs=[(query,chunks[i]) for i in candidate_idx]

    rerank_scores=reranker.predict(pairs)

    order=np.argsort(-rerank_scores)
    return order,candidate_idx,rerank_scores

print("\n=== RERANKED RESULTS (CrossEncoder) ===")

order,candidate_idx,rerank_scores= cross_encoding(ranking,query,chunks)
 
for rank,pos in enumerate(order[:5],start=1):
    idx=candidate_idx[pos]
    print(f"\nRank {rank} | Score: {rerank_scores[pos]:.4f} | Chunk idx: {idx}")
    print(chunks[idx][:300].replace("\n", " "), "...")















