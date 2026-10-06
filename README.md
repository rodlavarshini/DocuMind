# 📚 DocuMind

**DocuMind: An AI-Powered RAG-Based Document Intelligence and Question Answering System**

**Student:** RODLA VARSHINI  
**Roll No:** 25RH1A67R8

## What the project does

DocuMind is a complete RAG (Retrieval-Augmented Generation) application. A user uploads PDF documents, the application extracts their text, splits it into chunks, creates semantic embeddings, stores those embeddings in ChromaDB, retrieves the most relevant passages for a question, and uses a local Hugging Face Transformers model to generate an answer from the retrieved context.

### Main features

- PDF upload
- Page-level PDF text extraction using `pypdf`
- Overlapping text chunking
- Semantic embeddings with Sentence Transformers
- Persistent ChromaDB vector database
- Top-K semantic retrieval
- Local question answering with Hugging Face Transformers + PyTorch
- Source document and page display
- Streamlit web interface
- No external API key required by default

## Project structure

```text
my_project/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── .env
├── assets/
└── venv/
```

`data/` is created automatically at runtime for the ChromaDB index and is ignored by Git.

## Run locally

Python 3.12 is recommended.

### Windows

```powershell
cd my_project
py -3.12 -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

### macOS/Linux

```bash
cd my_project
python3.12 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

## Deploy with Streamlit Community Cloud

1. Create a GitHub repository and upload `app.py`, `requirements.txt`, `README.md`, `.gitignore`, and `assets/`.
2. Do **not** upload `venv/` or `.env`.
3. In Streamlit Community Cloud, choose the repository and `app.py` as the main file.
4. Deploy.
5. The first startup downloads the Sentence Transformer and `google/flan-t5-small` models, so the first launch can take longer.
6. Upload a text-based PDF and click **Index Documents**.
7. Ask a question about the uploaded document.

## Important deployment note

The application uses PyTorch, Sentence Transformers, Transformers, and ChromaDB, so it is considerably larger than a simple Streamlit demo. The free deployment environment may have memory/storage limitations. If model loading fails because of memory, the first thing to reduce is the generation model size or move generation to an external model/API later.

Scanned/image-only PDFs are not OCR'd by this version because OCR is not included in the requested dependency list. Such PDFs need selectable text to be indexed.

## Architecture

```text
             PDF Upload
                  ↓
           PDF Text Extraction
                  ↓
            Text Chunking
                  ↓
       Sentence Transformer
             Embeddings
                  ↓
             ChromaDB
                  ↓
          Semantic Retrieval
                  ↓
       Retrieved Document Context
                  ↓
      Hugging Face Transformers
          (FLAN-T5 Small)
                  ↓
          Grounded Answer
                  ↓
        Sources + Page Numbers
```

## Technology stack

- Python 3.12
- Streamlit
- pypdf
- Sentence Transformers
- ChromaDB
- Transformers
- PyTorch
- python-dotenv

## Requirements

The supplied package ranges are kept in `requirements.txt`.
