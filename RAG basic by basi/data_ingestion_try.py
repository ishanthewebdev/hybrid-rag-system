# import pdfplumber
# import pytesseract
# from PIL import ImageChops
# import re
# from huggingface_hub import InferenceClient
# import json
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# #checking for scanner image

# def is_scanned(page):
#     text=page.extract_text() or ''
#     return len(text.strip())<50

# def ocr_page(page):
#     # # img=page.to_image().original
#     # # return pytesseract.image_to_string(img)
#     # img = page.to_image().original
#     # text = pytesseract.image_to_string(img)
#     # print("RAW OCR OUTPUT (first 200 chars):")
#     # print(repr(text[:200]))
#     # return text
#     img = page.to_image(resolution=300).original

#     # DEBUG: save image to see what Tesseract is seeing
#     img.save("debug_page0.png")

#     print("Saved debug_page0.png")

#     text = pytesseract.image_to_string(img, lang="eng")
#     print("RAW OCR OUTPUT (first 200 chars):")
#     print(repr(text[:200]))
#     return text

# def clean_text(text):
#     # learn-\ning --> learning
#     # text=re.sub(r'-\n,',text)
#     text = re.sub(r"-\n", "", text)

    
#     # \r -> \n (safe newlines)
#     text = text.replace("\r", "\n")
#       # extra newlines limit to max 2
#     text = re.sub(r"\n{3,}", "\n\n", text)  

#     #stripping spaces
#     return text.strip()


# def load_pdf(path):
#     final_pages=[]

#     with pdfplumber.open(path) as pdf:
#         for i,page in enumerate(pdf.pages):
#             if is_scanned(page):
#                 print(f"Page {i}:OCR mode")
#                 raw=ocr_page(page)
#             else:
#                 print(f"Page{i}:Normal text")    
#                 raw=page.extract_text() or ""


#             cleaned=clean_text(raw)

#             final_pages.append(cleaned)
#     # result = load_pdf("scanned_test.pdf")
#     # print("FINAL RESULT LENGTH:", len(result))
#     # print("FIRST 500 CHARS:", repr(result[:500]))        
#     return "\n\n=== PAGE BREAK ===\n\n".join(final_pages)      
#     result = load_pdf("scanned_test.pdf")
#     print("FINAL RESULT LENGTH:", len(result))
#     print("FIRST 500 CHARS:", repr(result[:500]))

# text= load_pdf("sample-pdf-file.pdf")
# # result = load_pdf("scanned_test.pdf")
# print("FINAL RESULT LENGTH:", len(text))
# print(text[:2000])
        
  

# # HF client
# client = InferenceClient(
#     model="HuggingFaceH4/zephyr-7b-beta",
#     token="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"
# )

# # 1) PARAGRAPH SPLITTER
# def make_paragraphs(text):
#     print("\n========== STEP 1: MAKING PARAGRAPHS ==========")
#     paras = [p.strip() for p in text.split("\n\n") if p.strip()]
#     for i, p in enumerate(paras):
#         print(f"[{i}] {p[:80]}...")
#     return paras


# # 2) PROMPT BUILDER
# def build_simple_agentic_prompt(paragraphs):
#     numbered = "\n\n".join([f"[{i}] {p}" for i, p in enumerate(paragraphs)])
#     prompt = f"""
# You are an expert at splitting long text into meaningful sections.

# Below are numbered paragraphs from a document.

# Your job:
# 1. Combine related paragraphs into ONE chunk.
# 2. Each chunk must use CONTIGUOUS paragraph indices.
# 3. For each chunk give:
#    - "start": paragraph start index
#    - "end": paragraph end index
#    - "title": short 3-6 word title
#    - "summary": one sentence summary

# Output ONLY JSON list.

# PARAGRAPHS:
# {numbered}
# """
#     print("\n========== STEP 2: PROMPT SENT TO LLM ==========")
#     print(prompt)
#     return prompt


# # 3) ASK LLM
# def ask_llm(prompt):
#     print("\n========== STEP 3: LLM RAW OUTPUT ==========")
#     out = client.text_generation(prompt, max_new_tokens=400, temperature=0.2, do_sample=False)
#     print(out)
#     return out


# # 4) PROCESS JSON
# def get_plan(paragraphs):
#     prompt = build_simple_agentic_prompt(paragraphs)
#     raw = ask_llm(prompt)

#     try:
#         plan = json.loads(raw)
#     except:
#         plan = [{"start": 0, "end": len(paragraphs)-1, "title": "Full Doc", "summary": "All"}]

#     print("\n========== STEP 4: LLM CHUNK PLAN ==========")
#     print(json.dumps(plan, indent=2))
#     return plan


# # 5) FINAL CHUNK BUILDER
# def build_chunks(paragraphs, plan):
#     print("\n========== STEP 5: FINAL CHUNKS BUILT ==========")
#     final = []

#     for i, ch in enumerate(plan):
#         s = ch["start"]
#         e = ch["end"]
#         content = "\n\n".join(paragraphs[s:e+1])

#         chunk = {
#             "chunk_id": i,
#             "start": s,
#             "end": e,
#             "title": ch["title"],
#             "summary": ch["summary"],
#             "content": content
#         }

#         print(f"\n--- Chunk {i} ---")
#         print(json.dumps(chunk, indent=2))

#         final.append(chunk)

#     return final


# # 6) VISUALIZER
# # COLORS = ["\033[92m", "\033[93m", "\033[94m", "\033[95m", "\033[96m"]
# # RESET = "\033[0m"

# # def visualize_chunks(paragraphs, plan):
# #     print("\n========== STEP 6: VISUALIZATION ==========")
# #     for i, ch in enumerate(plan):
# #         color = COLORS[i % len(COLORS)]
# #         print(f"\n{color}--- Chunk {i}: {ch['title']} ---{RESET}")
# #         for idx in range(ch["start"], ch["end"]+1):
# #             print(color + f"[{idx}] {paragraphs[idx]}" + RESET)


# # MAIN PIPELINE
# def RGRAC_simple(text):
#     paras = make_paragraphs(text)
#     plan = get_plan(paras)
#     # visualize_chunks(paras, plan)
#     return build_chunks(paras, plan)

import pdfplumber
import pytesseract
from PIL import ImageChops
import re
from huggingface_hub import InferenceClient
import json
import os
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from sentence_transformers import SentenceTransformer
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util


pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# load_dotenv()
load_dotenv(dotenv_path=r"C:\Users\ishan\Desktop\langchain campusx\.env")

# ----------------------------------------
# PDF LOADING (OCR + CLEANING)
# ----------------------------------------
def is_scanned(page):
    text = page.extract_text() or ''
    return len(text.strip()) < 50


def ocr_page(page):
    img = page.to_image(resolution=300).original
    text = pytesseract.image_to_string(img, lang="eng")
    print("RAW OCR OUTPUT:", repr(text[:200]))
    return text


def clean_text(text):
    text = re.sub(r"-\n", "", text)
    text = text.replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_pdf(path):
    pages = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages):
            if is_scanned(page):
                print(f"Page {i}: OCR mode")
                raw = ocr_page(page)
            else:
                print(f"Page {i}: Normal text")
                raw = page.extract_text() or ""

            cleaned = clean_text(raw)
            pages.append(cleaned)

    return "\n\n=== PAGE BREAK ===\n\n".join(pages)


text = load_pdf("small_rag_test.pdf")
print("PDF LOADED | LENGTH:", len(text))


# ----------------------------------------
# HUGGINGFACE CLIENT
# ----------------------------------------
# client = InferenceClient(
#     model="HuggingFaceH4/zephyr-7b-beta",
#     token="YOUR_HF_TOKEN"
# )
llm=HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    # temperature=0.1
    )

model = ChatHuggingFace(llm=llm)


# ----------------------------------------
# UTILS
# ----------------------------------------
# def extract_json(text):
#     start = text.find('[')
#     end = text.rfind(']')
#     if start != -1 and end != -1:
#         return text[start:end + 1]
#     return None
def extract_json(text):
    text = str(text)
    start = text.find('[')
    end = text.rfind(']')
    if start != -1 and end != -1:   
        return text[start:end+1]
    return None


# ----------------------------------------
# STEP 1: PARAGRAPH SPLIT
# ----------------------------------------
def make_paragraphs(text):
    print("\n=== STEP 1: PARAGRAPHS ===")
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]

    # LLM-friendly (truncate)
    paras = [p[:300] for p in paras]

    for i, p in enumerate(paras):
        print(f"[{i}] {p[:100]}...")

    return paras


# ----------------------------------------
# STEP 2: PROMPT
# ----------------------------------------
def build_simple_agentic_prompt(paragraphs):
    numbered = "\n\n".join([f"[{i}] {p}" for i, p in enumerate(paragraphs)])

    prompt = f"""
You are an expert at splitting long text into meaningful sections.

Below are numbered paragraphs from a document.

Your job:
1. Combine related paragraphs into ONE chunk.
2. Each chunk must use CONTIGUOUS paragraph indices.
3. For each chunk give:
   - "start"
   - "end"
   - "title"
   - "summary"

Output ONLY JSON array.

PARAGRAPHS:
{numbered}
"""
    print("\n=== STEP 2: PROMPT ===")
    print(prompt)
    return prompt


# ----------------------------------------
# STEP 3: ASK LLM
# ----------------------------------------
# def ask_llm(prompt):
#     # print("\n=== STEP 3: RAW LLM OUTPUT ===")
#     print("\n========== STEP 3: LLM RAW OUTPUT ==========")

#     response = model.chat_completion(
#         messages=[
#             {"role": "user", "content": prompt}
#         ],
#         max_tokens=400,
#         temperature=0.2,
#     )

#     # HF chat_completion ka output OpenAI jaisa hota hai:
#     # response.choices[0].message.content
#     out = response.choices[0].message.content
#     print(out)
#     return out
#     # out = client.text_generation(prompt, max_new_tokens=400,
#     #                              temperature=0.2, do_sample=False)
#     # print(out)
#     # return out
def ask_llm(prompt):
    print("\n========== STEP 3: LLM RAW OUTPUT ==========")

    out = model.invoke(prompt) # same as "human prompt"
    print(out)
    return out.content



# ----------------------------------------
# STEP 4: PROCESS JSON
# ----------------------------------------
# def get_plan(paragraphs):
#     prompt = build_simple_agentic_prompt(paragraphs)
#     raw = ask_llm(prompt)

#     extracted = extract_json(raw)

#     if extracted:
#         plan = json.loads(extracted)
#     else:
#         plan = [{"start": 0, "end": len(paragraphs)-1,
#                  "title": "Full Doc", "summary": "All"}]

#     print("\n=== STEP 4: PLAN ===")
#     print(json.dumps(plan, indent=2))
#     return plan

def get_plan(paragraphs):
    prompt = build_simple_agentic_prompt(paragraphs)
    raw = ask_llm(prompt)

    extracted = extract_json(raw)

    if extracted:
        try:
            plan = json.loads(extracted)
        except json.JSONDecodeError as e:
            print("\n[WARN] JSON parsing failed, falling back to single chunk.")
            print("[DEBUG] Extracted JSON string was:\n", extracted[:500], "...\n")
            plan = [{
                "start": 0,
                "end": len(paragraphs) - 1,
                "title": "Full Document",
                "summary": "Single chunk fallback (LLM JSON parse failed)."
            }]
    else:
        print("\n[WARN] No JSON array found in LLM output, using fallback plan.")
        plan = [{
            "start": 0,
            "end": len(paragraphs) - 1,
            "title": "Full Document",
            "summary": "Single chunk fallback (no JSON in output)."
        }]

    print("\n=== STEP 4: PLAN ===")
    print(json.dumps(plan, indent=2))
    return plan

# ----------------------------------------
# STEP 5: BUILD CHUNKS
# ----------------------------------------
def build_chunks(paragraphs, plan):
    print("\n=== STEP 5: BUILD CHUNKS ===")
    final = []

    for i, ch in enumerate(plan):
        s = ch["start"]
        e = ch["end"]
        content = "\n\n".join(paragraphs[s:e+1])

        chunk = {
            "chunk_id": i,
            "start": s,
            "end": e,
            "title": ch["title"],
            "summary": ch["summary"],
            "content": content
        }

        print(f"\n--- CHUNK {i} ---")
        print(json.dumps(chunk, indent=2))

        final.append(chunk)

    return final


# ----------------------------------------
# MASTER PIPELINE
# ----------------------------------------
def RGRAC_simple(text):
    paras = make_paragraphs(text)
    plan = get_plan(paras)

    print("\n=== FINAL PLAN ===")
    print(plan)

    return build_chunks(paras, plan)

# ----------------------------------------
# EMBEDDING MODEL (local, free)
# ----------------------------------------
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def build_embedding_index(chunks,query):
    """
    Har chunk ke 'content' ko embedding me convert karta hai
    aur ek numpy array + original chunks return karta hai.
    """
    texts = [c["content"] for c in chunks]
    tokenized_docs = [texts.split() for doc in texts]
    bm25 = BM25Okapi(tokenized_docs)
    bm25_scores = bm25.get_scores(query.split())

    

    # BGE recommendation: normalize embeddings for cosine similarity
    embeddings = embed_model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True
    )
    emb_query = model.encode(query, convert_to_tensor=True)
    embeddings = np.array(embeddings)

    dense_scores = util.cos_sim(emb_query, embeddings).gpu().tolist()[0]
      

    # numpy array me convert
    # embeddings = np.array(embeddings)
    hybrid_scores = np.array(dense_scores) + 0.5 * np.array(bm25_scores)
    ranking = sorted(list(enumerate(hybrid_scores)), key=lambda x: x[1], reverse=True)
    print("\n=== HYBRID RESULTS ===")
    for idx, score in ranking:
     print(f"{score:.4f}  →  {texts[idx]}")
       

    return embeddings, texts, chunks
# ----------------------------------------
# RUN THE PIPELINE
# ----------------------------------------
if __name__ == "__main__":
    chunks = RGRAC_simple(text)
    print("\n\n=== FINAL CHUNKS READY FOR EMBEDDING ===")
    for c in chunks:
        print("\n", c)
    query="what is rag"    
    embeddings, texts, chunks = build_embedding_index(chunks,query)

    print("\nEmbedding index shape:", embeddings.shape)
