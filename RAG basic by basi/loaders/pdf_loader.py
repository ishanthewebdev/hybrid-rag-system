import pdfplumber
import pytesseract
import re

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def is_scanned(page):
    text = page.extract_text() or ""
    return len(text.strip()) < 50

def ocr_page(page):
    img = page.to_image(resolution=300).original
    return pytesseract.image_to_string(img, lang="eng")

def clean_text(text):
    text = re.sub(r"-\n", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def load_pdf(path):
    pages = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            raw = ocr_page(page) if is_scanned(page) else page.extract_text() or ""
            pages.append(clean_text(raw))
    return "\n\n".join(pages)
