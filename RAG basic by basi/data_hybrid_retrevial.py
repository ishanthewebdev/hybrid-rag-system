from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util,CrossEncoder
# from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
import json

import pdfplumber
import pytesseract
from PIL import ImageChops
import re
from huggingface_hub import InferenceClient
import json
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
from langchain_huggingface import HuggingFaceEmbeddings ,ChatHuggingFace,HuggingFaceEndpoint

from dotenv import load_dotenv
#
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
# from langchain_community.llms import Ollama
from langchain_ollama import OllamaLLM

load_dotenv()
llm=HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    # temperature=0.1
    )
# llm = OllamaLLM(model="llama3")
model2 = ChatHuggingFace(llm=llm)
# model2=llm
# model2 = OllamaLLM(model="llama3")


#checking for scanner image


def is_scanned(page):
    text=page.extract_text() or ''
    return len(text.strip())<50

def ocr_page(page):
    # # img=page.to_image().original
    # # return pytesseract.image_to_string(img)
    # img = page.to_image().original
    # text = pytesseract.image_to_string(img)
    # print("RAW OCR OUTPUT (first 200 chars):")
    # print(repr(text[:200]))
    # return text
    img = page.to_image(resolution=300).original

    # DEBUG: save image to see what Tesseract is seeing
    img.save("debug_page0.png")

    print("Saved debug_page0.png")

    text = pytesseract.image_to_string(img, lang="eng")
    print("RAW OCR OUTPUT (first 200 chars):")
    print(repr(text[:200]))
    return text

def clean_text(text):
    # learn-\ning --> learning
    # text=re.sub(r'-\n,',text)
    text = re.sub(r"-\n", "", text)

    
    # \r -> \n (safe newlines)
    text = text.replace("\r", "\n")
      # extra newlines limit to max 2
    text = re.sub(r"\n{3,}", "\n\n", text)  

    #stripping spaces
    return text.strip()


def load_pdf(path):
    final_pages=[]

    with pdfplumber.open(path) as pdf:
        for i,page in enumerate(pdf.pages):
            if is_scanned(page):
                print(f"Page {i}:OCR mode")
                raw=ocr_page(page)
            else:
                print(f"Page{i}:Normal text")    
                raw=page.extract_text() or ""


            cleaned=clean_text(raw)

            final_pages.append(cleaned)
    # result = load_pdf("scanned_test.pdf")
    # print("FINAL RESULT LENGTH:", len(result))
    # print("FIRST 500 CHARS:", repr(result[:500]))        
    return "\n\n=== PAGE BREAK ===\n\n".join(final_pages)      
    result = load_pdf("scanned_test.pdf")
    print("FINAL RESULT LENGTH:", len(result))
    print("FIRST 500 CHARS:", repr(result[:500]))

text= load_pdf("ml2.pdf")
# result = load_pdf("scanned_test.pdf")
print("FINAL RESULT LENGTH:", len(text))
print(text[:2000])

splitter=RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=0,

)
chunks=splitter.split_text(text)
docs = [p.strip() for p in text.split("\n\n") if p.strip()]
print(f"\nTotal DOCS (paragraphs): {len(docs)}")

#for better query writing
def rewrite_query(original_query: str, model2, n_variants: int = 1) -> str:
    """
    User ki query ko thoda zyada clear, detailed
    search query me convert karta hai.
    Agar kuch bhi gadbad hui, original query hi use karega.
    """
    prompt = f"""
You are a search query rewriter.

User query: "{original_query}"

Task:
- Rewrite this as a clearer, more detailed search query.
- Keep the SAME meaning and topic.
- Expand abbreviations and add important related words.
- Do NOT change the intent.

Return ONLY a JSON object in this exact format:
{{"rewritten": "your rewritten query here"}}
"""

    resp = model2.invoke(prompt)
    # text = resp.content.strip()
    text=resp
    print("\n[DEBUG] Raw rewrite LLM output:\n", text, "\n")

    # JSON extract
    try:
        # kabhi-kabhi extra text se pehle/baad aa sakta hai, isliye braces se cut karte hain
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            raise ValueError("No JSON object found")

        json_str = text[start:end+1]
        data = json.loads(json_str)

        rewritten = str(data.get("rewritten", "")).strip()
    except Exception as e:
        print("[WARN] Failed to parse rewritten query, using original. Error:", e)
        return original_query

    # Basic sanity checks
    if (
        not rewritten or
        len(rewritten) < 5 or            # bahut chhota
        "[" in rewritten or
        "]" in rewritten or
        rewritten.upper() == rewritten   # pure caps = suspicious
    ):
        print("[WARN] Rewritten query looks invalid, using original instead.")
        return original_query

    return rewritten
 
user_query = input("ask question about the pdf : ")

rewritten_query = rewrite_query(user_query, model2)

print("\nOriginal query: ", user_query)
print("Rewritten query:", rewritten_query)

# Ab se har jagah isi rewritten ko use karo
query = user_query

# --- Sparse: BM25 ---
tokenized_chunks = [c.split() for c in chunks]
bm25 = BM25Okapi(tokenized_chunks)

# tokenized_docs = [doc.split() for doc in docs]
bm25 = BM25Okapi(tokenized_chunks)
bm25_scores = bm25.get_scores(query.split())

# --- Dense: Embeddings ---
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
emb_docs = model.encode(chunks, convert_to_tensor=True)
emb_query = model.encode(query, convert_to_tensor=True)
dense_scores = util.cos_sim(emb_query, emb_docs).cpu().tolist()[0]

# --- Hybrid Score ---
import numpy as np
hybrid_scores = np.array(dense_scores) + 0.5 * np.array(bm25_scores)

# --- Sorting ---
# ranking = sorted(list(enumerate(hybrid_scores)), key=lambda x: x[1], reverse=True)
# ranking = np.argsort(-hybrid_scores)

# print("\n=== HYBRID RESULTS ===")
# for idx, score in ranking:
#     print(f"{score:.4f}  →  {text[idx]}")
ranking = np.argsort(-hybrid_scores)  # descending indices

print("\n=== HYBRID RESULTS ===")
for idx in ranking[:5]:   # top-5
    print(f"Score: {hybrid_scores[idx]:.4f}")
    print("DOC:", chunks[idx][:300].replace("\n", " "), "...")


# -----------------------------
# RERANKING WITH CROSSENCODER
# -----------------------------

# 1) Pehle top-N candidates hybrid se lo
top_n = 10
candidate_idx = ranking[:top_n]   # ranking np.argsort se aaya hai

# 2) CrossEncoder model load karo (zyada accha hai global pe load karna, but abhi simple rakhte hain)
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# 3) Query + chunk pairs banao
pairs = [(query, chunks[i]) for i in candidate_idx]

# 4) Reranker se scores nikaalo
rerank_scores = reranker.predict(pairs)   # shape: (top_n,)

# 5) Scores ke basis pe order nikaalo (descending)
order = np.argsort(-rerank_scores)

print("\n=== RERANKED RESULTS (CrossEncoder) ===")
for rank, pos in enumerate(order[:5], start=1):   # final top-5
    idx = candidate_idx[pos]  # original chunk index

    print(f"\nRank {rank} | Score: {rerank_scores[pos]:.4f} | Chunk idx: {idx}")
    print(chunks[idx][:300].replace("\n", " "), "...")


# top_k = 5
# top_chunk_indices = ranking[:top_k]
# top_chunks = [chunks[i] for i in top_chunk_indices]

#pehle top chunks manually lerhe the ab use thoda guard kar rhe hai for better result like agar context shi nhi hai to mat reply karo like that,if score is too low and all
# chunks = list[str]
# hybrid_scores = numpy array, same length as chunks

import numpy as np

def select_top_chunks_with_guardrails(chunks, hybrid_scores, 
                                      max_k=5, min_score=0.2):
    """
    chunks: list[str]
    hybrid_scores: np.array of floats
    max_k: max chunks to use
    min_score: agar top score isse kam hai -> no good match
    """
    if len(chunks) == 0:
        return []

    # scores high to low
    ranking = np.argsort(-hybrid_scores)

    # best score
    best_score = hybrid_scores[ranking[0]]
    print(f"Best hybrid score: {best_score:.4f}")

    # agar best hi bohot low -> return empty
    if best_score < min_score:
        print("[GUARD] Best score too low, no reliable match.")
        return []

    selected = []
    for idx in ranking:
        if len(selected) >= max_k:
            break
        ch = chunks[idx].strip()
        if not ch:
            continue
        selected.append(ch)

    return selected

def remove_duplicate_chunks(chunks):
    """
    Simple dedup: agar chunk already kisi added chunk ka
    substring hai ya same hai -> skip.
    """
    unique = []
    for ch in chunks:
        ch_clean = " ".join(ch.split())  # normalize spaces
        skip = False
        for u in unique:
            if ch_clean in u or u in ch_clean:
                skip = True
                break
        if not skip:
            unique.append(ch_clean)
    return unique


top_chunks = select_top_chunks_with_guardrails(
    chunks,
    hybrid_scores,
    max_k=5,
    min_score=0.2,   # tune kar sakte ho
)

top_chunks = remove_duplicate_chunks(top_chunks) #unique elements ko alag kar rhe hai and duplicae chunks ko hata rhe hai
print(f"Using {len(top_chunks)} unique chunks after dedup.")

def build_context(top_chunks, max_chars=1500):
    """
    top_chunks: list[str] - best chunks (already reranked or hybrid sorted)
    max_chars: zyada lamba context avoid karne ke liye hard limit

    Returns: single string 'context'
    """
    parts = []
    total = 0

    for ch in top_chunks:
        text = ch.strip()
        if not text:
            continue

        # Agar yeh pura chunk add karne se limit cross ho rahi:
        if total + len(text) > max_chars:
            remaining = max_chars - total
            if remaining > 100:  # thoda meaningful part to le hi le
                parts.append(text[:remaining])
            break

        parts.append(text)
        total += len(text)

    # Chunks ko separator ke saath join karo (sirf readability ke liye)
    context = "\n\n---\n\n".join(parts)
    return context


context = build_context(top_chunks)
print("\n=== CONTEXT ===\n")
print(context[:1000])


def answer_with_rag(query, context):
    """
    Query + retrieved context LLM ko dekar
    final answer generate karta hai.
    """
    if not context or len(context.strip()) < 50:
        return "I don't know based on the provided document." #this is used for guardrails or extra protection regarding what llm will return so better return idk rather than hallucinating

    prompt = f"""
You are a helpful AI assistant.

Use ONLY the information in the CONTEXT below to answer the QUESTION.
If the answer is not in the context, say exactly:
"I don't know based on the provided document."

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:"""

    response = model2.invoke(prompt)


    return response.content
    # return response
def validate_answer(query, context, answer, model2):
    """
    LLM se poochte hain:
    - kya MODEL ANSWER context se supported hai?
    - kya answer really user ke question ka jawab de raha hai?

    Return: dict
      {
        "supported": bool,
        "relevance": float (0-1),
        "reason": "short explanation"
      }
    """
    prompt = f"""
You are a strict RAG answer judge.

You are given:
1) USER QUESTION
2) CONTEXT (retrieved from a document)
3) MODEL ANSWER (generated using that context)

Your job:
- Check if the MODEL ANSWER is actually supported by the CONTEXT.
- Check if it really answers the USER QUESTION.
- If the answer adds extra information which is NOT clearly in the context, mark it as unsupported.

Return ONLY a JSON object with keys:
- "supported": true or false
- "relevance": a number between 0 and 1 (1 = perfectly answers, 0 = not related)
- "reason": short one-sentence explanation

USER QUESTION:
{query}

CONTEXT:
{context}

MODEL ANSWER:
{answer}
"""

    resp = model2.invoke(prompt)
    # text = resp.content.strip()
    text=resp

    try:
        result = json.loads(text)
    except Exception:
        # agar JSON parse fail ho gaya -> safe default
        result = {
            "supported": False,
            "relevance": 0.0,
            "reason": "Could not parse validator output."
        }

    # type safety
    result["supported"] = bool(result.get("supported", False))
    try:
        result["relevance"] = float(result.get("relevance", 0.0))
    except Exception:
        result["relevance"] = 0.0

    result["reason"] = str(result.get("reason", ""))
    return resp





# 4) Get final answer from LLM using RAG
# final_answer = answer_with_rag(query, context)

raw_answer = answer_with_rag(query, context)

print("\n=== RAW RAG ANSWER ===\n")
print(raw_answer)

# 2) ab judge ko bhejo
verdict = validate_answer(query, context, raw_answer, model2)

print("\n=== VALIDATION VERDICT ===")
print(verdict)

MIN_RELEVANCE = 0.6

if (not verdict["supported"]) or (verdict["relevance"] < MIN_RELEVANCE):
    final_answer = "I don't know based on the provided document."
else:
    final_answer = raw_answer

print("\n=== FINAL RAG ANSWER ===\n")
print(final_answer)

#ek aur step hai jisme prompot acha likhte hai to get good accurate answer
#now llm to dhair sare chunks dene se acha we will  form sentences so that llm will give good answers ,amybe summary of chunks etc methos

# from sentence_transformers import util

# def build_compressed_context(query, chunks,model,emb_query, 
#                              max_chars=1500, top_sent_per_chunk=2):
#     """
#     query: user question
#     chunks: list[str] - cleaned chunks
#     embed_model: SentenceTransformer model
#     max_chars: context length limit
#     top_sent_per_chunk: each chunk se top k sentences

#     Returns: compressed context string
#     """
#     compressed_parts = []

#     # Encode query once
   

#     for chunk in chunks:
#         # 1. sentence split
#         sentences = [s.strip() for s in chunk.split(". ") if s.strip()]
#         if not sentences:
#             continue

#         # 2. embed all sentences in chunk
#         sent_embs = model.encode(
#             sentences,
#             convert_to_tensor=True,
#             normalize_embeddings=True
#         )

#         # 3. similarity scores
#         scores = util.cos_sim(emb_query, sent_embs)[0].cpu().tolist()

#         # 4. pick top sentences

#         top_idx = sorted(
#             range(len(scores)), 
#             key=lambda i: scores[i], 
#             reverse=True
#         )[:top_sent_per_chunk]

#         best_sentences = [sentences[i] for i in top_idx]
#         compressed_parts.extend(best_sentences)

#         # 5. stop if max_chars reached
#         if sum(len(s) for s in compressed_parts) > max_chars:
#             break

#     # final context (small + strong)
#     return "\n".join(compressed_parts)


# # embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# context = build_compressed_context(
#     query=query,
#     chunks=top_chunks,        # reranked or top_k chunks
#     model=model,
#     emb_query=emb_query,
#     max_chars=1500,
#     top_sent_per_chunk=2
# )

# print("\n=== COMPRESSED CONTEXT ===\n")
# print(context)
