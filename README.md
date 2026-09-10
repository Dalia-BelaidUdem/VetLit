# VetLit-RAG

A Retrieval-Augmented Generation (RAG) pipeline that answers domain-specific questions over open-access veterinary research literature, fully local and free to run.

## What it does

1. **Downloads papers** from PubMed Central (PMC) using the NCBI E-utilities API — no PDFs, no scraping, just open-access full-text XML converted to plain text.
2. **Ingests and indexes** the text into a local ChromaDB vector store using `sentence-transformers/all-MiniLM-L6-v2` embeddings (runs on CPU, no GPU needed).
3. **Answers questions** by retrieving the 5 most relevant chunks and passing them as grounded context to an LLM via the Groq API.
4. **Evaluates** retrieval and answer quality with two custom metrics: context hit rate (keyword match in retrieved chunks) and faithfulness (fraction of answer sentences grounded in retrieved context).

**Evaluation results on 5 domain questions (mastitis, FT-MIR spectroscopy, prevalence, clustering, body condition score):**
- Context hit rate: **80%** (4/5)
- Average faithfulness: **88%**

## Stack

| Component | Tool |
|---|---|
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local, free) |
| Vector store | ChromaDB (local persistence) |
| LLM | `openai/gpt-oss-20b` via Groq API (free tier) |
| Framework | LangChain |
| Paper source | PubMed Central E-utilities XML API |

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env and add your Groq API key (free at console.groq.com)
```

## Usage

**Step 1 — Download papers from PubMed Central**
```bash
python -X utf8 download_papers.py
```
Papers are saved as `.txt` files in `papers/`. Edit the search queries in the script to target your topic.

**Step 2 — Ingest into vector store**
```bash
python -X utf8 ingest.py
```
Chunks and embeds all files in `papers/`, stores vectors in `data/chroma/`.

**Step 3 — Ask a question**
```bash
python -X utf8 query.py "What are the main risk factors for clinical mastitis in dairy cows?"
```
Returns a cited answer grounded in the retrieved chunks.

**Step 4 — Run evaluation**
```bash
python -X utf8 evaluate.py
```
Runs 5 test questions and saves results to `data/eval_results.csv`.

> **Note:** The `-X utf8` flag is required on Windows to avoid encoding errors.

## Architecture

```
PubMed Central (XML API)
     ↓  download_papers.py  →  papers/*.txt
     ↓  ingest.py
     ↓  RecursiveCharacterTextSplitter  (chunk: 800 chars, overlap: 150)
     ↓  all-MiniLM-L6-v2  (embed locally)
     ↓  ChromaDB  (persist to data/chroma/)
     ↓  query.py
     ↓  similarity search  (top-5 chunks)
     ↓  Groq LLM  (generate grounded answer with citations)
     ↓  evaluate.py
     ↓  context hit rate + faithfulness score
```

## Project structure

```
vetlit-rag/
├── download_papers.py   # fetch full-text articles from PMC
├── ingest.py            # chunk, embed, store in ChromaDB
├── query.py             # retrieve + generate answers
├── evaluate.py          # compute hit rate and faithfulness
├── requirements.txt
├── .env.example
├── papers/              # downloaded .txt files (gitignored)
└── data/
    ├── chroma/          # vector store (gitignored)
    └── eval_results.csv # evaluation output (gitignored)
```

## Author

Dalia Belaid — [daliabelaid@gmail.com](mailto:daliabelaid@gmail.com)
