"""
ingest.py — Load PDF papers → chunk → embed → store in ChromaDB
Usage: python ingest.py
"""

import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

PAPERS_DIR = Path("papers")
CHROMA_DIR = Path("data/chroma")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


def load_papers():
    docs = []
    files = list(PAPERS_DIR.glob("*.pdf")) + list(PAPERS_DIR.glob("*.txt"))
    if not files:
        print("No files found in ./papers/ — add PDFs or .txt files and re-run.")
        return []
    for f in files:
        print(f"  Loading {f.name}...")
        try:
            if f.suffix == ".pdf":
                loader = PyPDFLoader(str(f))
            else:
                loader = TextLoader(str(f), encoding="utf-8")
            pages = loader.load()
            for page in pages:
                page.metadata["source"] = f.name
            docs.extend(pages)
        except Exception as e:
            print(f"    Warning: could not load {f.name} — {e}")
    print(f"  >> {len(docs)} documents loaded from {len(files)} files")
    return docs


def chunk(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"  >> {len(chunks)} chunks created")
    return chunks


def embed_and_store(chunks):
    print(f"  Loading embedding model ({EMBED_MODEL})...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    print("  Storing in ChromaDB...")
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
        collection_name="vetlit",
    )
    print(f"  >> {db._collection.count()} vectors stored in {CHROMA_DIR}")
    return db


if __name__ == "__main__":
    print("\n=== VetLit-RAG Ingestion Pipeline ===\n")
    docs = load_papers()
    if not docs:
        exit(1)
    chunks = chunk(docs)
    embed_and_store(chunks)
    print("\nDone. Run query.py to ask questions.\n")
