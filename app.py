import re
from pathlib import Path

import faiss
import fitz
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="PinPoint AI", page_icon="📌", layout="wide")
st.title("📌 PinPoint AI")
st.caption("PDF Understanding, Semantic Search and RAG-ready Document Retrieval")

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_embedding_model()

def clean_text(text):
    return re.sub(r"\s+", " ", text).strip()

def chunk_text(text, chunk_size=900, overlap=150):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

uploaded = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded:
    doc = fitz.open(stream=uploaded.read(), filetype="pdf")

    chunks = []
    for page_num, page in enumerate(doc, start=1):
        text = clean_text(page.get_text("text"))
        for chunk_id, chunk in enumerate(chunk_text(text)):
            if chunk.strip():
                chunks.append({
                    "page": page_num,
                    "chunk_id": chunk_id,
                    "text": chunk
                })

    if not chunks:
        st.warning("No extractable text was found. Scanned PDFs may require OCR.")
    else:
        texts = [x["text"] for x in chunks]
        embeddings = model.encode(texts, normalize_embeddings=True).astype("float32")

        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)

        st.success(f"Processed {len(doc)} pages into {len(chunks)} chunks.")

        query = st.text_input("Ask or search within the document")

        if query:
            q = model.encode([query], normalize_embeddings=True).astype("float32")
            scores, ids = index.search(q, min(5, len(chunks)))

            st.subheader("Most relevant document sections")
            for score, idx in zip(scores[0], ids[0]):
                item = chunks[int(idx)]
                with st.expander(f"Page {item['page']} · Similarity {score:.3f}"):
                    st.write(item["text"])

st.markdown("---")
st.caption(
    "Core semantic retrieval demo. A generative-model layer can be added for grounded RAG answers, "
    "summaries, stories, scene generation and narration."
)
