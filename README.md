# VetLit-RAG

Retrieval-Augmented Generation pipeline over veterinary and animal health research literature.

Built with LangChain, Hugging Face sentence-transformers, ChromaDB, and RAGAS.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # add your OpenAI API key
```

## Usage

**Step 1 — Add papers**
Put PDF papers in the `papers/` folder.
Good sources: PubMed, Journal of Dairy Science, Preventive Veterinary Medicine.

**Step 2 — Ingest**
```bash
python ingest.py
```

**Step 3 — Query**
```bash
python query.py "What factors predict productive longevity in dairy cows?"
```

**Step 4 — Evaluate**
```bash
python evaluate.py
```

## Architecture

```
papers/*.pdf
     ↓  PyPDF (load)
     ↓  RecursiveCharacterTextSplitter (chunk: 800 tokens, overlap: 150)
     ↓  HuggingFace all-MiniLM-L6-v2 (embed)
     ↓  ChromaDB (store)
     ↓  similarity search (top-5)
     ↓  GPT-4o-mini (generate with grounded context)
     ↓  RAGAS (evaluate faithfulness, relevancy, precision, recall)
```
