import os
import re
from io import BytesIO
from pathlib import Path

import chromadb
import streamlit as st
import torch
from dotenv import load_dotenv
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

load_dotenv()

APP_TITLE = "DocuMind"
EMBED_MODEL = os.getenv("DOCUMIND_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
LLM_MODEL = os.getenv("DOCUMIND_LLM_MODEL", "google/flan-t5-small")
TOP_K = int(os.getenv("DOCUMIND_TOP_K", "5"))
CHUNK_SIZE = int(os.getenv("DOCUMIND_CHUNK_SIZE", "900"))
CHUNK_OVERLAP = int(os.getenv("DOCUMIND_CHUNK_OVERLAP", "150"))
DB_PATH = Path(os.getenv("DOCUMIND_CHROMA_DIR", "data/chroma"))
DB_PATH.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title=APP_TITLE, page_icon="📚", layout="wide")


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def extract_pages(file_bytes: bytes, filename: str):
    reader = PdfReader(BytesIO(file_bytes))
    pages = []
    for page_no, page in enumerate(reader.pages, start=1):
        text = clean_text(page.extract_text())
        if text:
            pages.append({"filename": filename, "page": page_no, "text": text})
    return pages


def make_chunks(pages):
    chunks = []
    for page in pages:
        text = page["text"]
        start = 0
        number = 0
        while start < len(text):
            end = min(start + CHUNK_SIZE, len(text))
            if end < len(text):
                candidates = [text.rfind(". ", start, end), text.rfind("? ", start, end), text.rfind("! ", start, end), text.rfind(" ", start, end)]
                boundary = max(candidates)
                if boundary > start + CHUNK_SIZE // 2:
                    end = boundary + 1
            piece = text[start:end].strip()
            if piece:
                chunks.append({
                    "id": f"{page['filename']}__p{page['page']}__c{number}",
                    "text": piece,
                    "filename": page["filename"],
                    "page": page["page"],
                })
                number += 1
            if end >= len(text):
                break
            start = max(end - CHUNK_OVERLAP, start + 1)
    return chunks


@st.cache_resource(show_spinner="Loading embedding model...")
def get_embedder():
    return SentenceTransformer(EMBED_MODEL)


@st.cache_resource(show_spinner="Loading local AI model...")
def get_generator():
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return tokenizer, model, device


@st.cache_resource
def get_collection():
    client = chromadb.PersistentClient(path=str(DB_PATH))
    return client.get_or_create_collection(
        name="documind",
        metadata={"hnsw:space": "cosine"},
    )


def index_chunks(chunks):
    collection = get_collection()
    model = get_embedder()
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, normalize_embeddings=True).tolist()
    collection.upsert(
        ids=[c["id"] for c in chunks],
        documents=texts,
        embeddings=embeddings,
        metadatas=[{"filename": c["filename"], "page": c["page"]} for c in chunks],
    )
    return len(chunks)


def search(question):
    collection = get_collection()
    count = collection.count()
    if count == 0:
        return []
    model = get_embedder()
    query_embedding = model.encode([question], normalize_embeddings=True).tolist()[0]
    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(TOP_K, count),
        include=["documents", "metadatas", "distances"],
    )
    docs = result["documents"][0]
    metas = result["metadatas"][0]
    distances = result["distances"][0]
    return [
        {"text": d, "filename": m["filename"], "page": m["page"], "distance": float(x)}
        for d, m, x in zip(docs, metas, distances)
    ]


def generate_answer(question, sources):
    tokenizer, model, device = get_generator()
    context = "\n\n".join(
        f"Source: {s['filename']} page {s['page']}\n{s['text']}" for s in sources
    )
    prompt = f"""Answer the question using ONLY the supplied document context.
If the context does not contain the answer, say that the answer was not found in the uploaded documents.
Do not invent facts.

DOCUMENT CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=180, num_beams=4)
    return tokenizer.decode(output[0], skip_special_tokens=True).strip()


def clear_index():
    client = chromadb.PersistentClient(path=str(DB_PATH))
    try:
        client.delete_collection("documind")
    except Exception:
        pass
    st.cache_resource.clear()


st.title("📚 DocuMind")
st.markdown("### An AI-Powered RAG-Based Document Intelligence and Question Answering System")
st.write("Upload PDF documents, build a semantic knowledge base, and ask questions grounded in those documents.")

with st.sidebar:
    st.header("📄 Documents")
    uploads = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

    if st.button("🔎 Index Documents", type="primary", use_container_width=True):
        if not uploads:
            st.warning("Upload at least one PDF first.")
        else:
            total = 0
            progress = st.progress(0)
            for i, uploaded in enumerate(uploads):
                try:
                    pages = extract_pages(uploaded.getvalue(), uploaded.name)
                    chunks = make_chunks(pages)
                    if chunks:
                        total += index_chunks(chunks)
                        st.success(f"{uploaded.name}: {len(chunks)} chunks indexed")
                    else:
                        st.warning(f"{uploaded.name}: no selectable text found")
                except Exception as exc:
                    st.error(f"{uploaded.name}: {exc}")
                progress.progress((i + 1) / len(uploads))
            st.info(f"Total chunks added/updated: {total}")

    collection = get_collection()
    st.metric("Indexed chunks", collection.count())

    if st.button("🗑️ Clear Knowledge Base", use_container_width=True):
        clear_index()
        st.success("Knowledge base cleared.")
        st.rerun()

    st.divider()
    st.caption("Models are downloaded the first time they are used. For deployment, allow enough memory for PyTorch and the models.")

st.divider()

left, right = st.columns([2, 1])
with left:
    st.subheader("💬 Ask your documents")
    question = st.text_area("Question", placeholder="Example: What are the main objectives mentioned in the document?", height=110)
    ask = st.button("Ask DocuMind", type="primary")

with right:
    st.subheader("How it works")
    st.markdown("**1. Upload** → **2. Extract** → **3. Embed** → **4. Retrieve** → **5. Generate**")
    st.caption("RAG = Retrieval-Augmented Generation")

if ask:
    if not question.strip():
        st.warning("Enter a question.")
    elif get_collection().count() == 0:
        st.warning("Index a PDF before asking questions.")
    else:
        with st.spinner("Retrieving relevant passages and generating answer..."):
            try:
                sources = search(question.strip())
                if not sources:
                    st.warning("No relevant document content was found.")
                else:
                    answer = generate_answer(question.strip(), sources)
                    st.subheader("🧠 Answer")
                    st.write(answer)

                    st.subheader("📚 Sources")
                    for i, source in enumerate(sources, 1):
                        with st.expander(f"Source {i} · {source['filename']} · Page {source['page']}"):
                            st.write(source["text"])
                            st.caption(f"ChromaDB cosine distance: {source['distance']:.4f}")
            except Exception as exc:
                st.error(f"The AI pipeline could not complete the request: {exc}")
                st.info("If this is the first run, make sure the deployment has internet access to download the Hugging Face models.")
