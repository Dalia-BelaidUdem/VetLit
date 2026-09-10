"""
query.py - Ask a question against the ingested papers
Usage: python query.py "What factors predict productive longevity in dairy cows?"
"""

import sys
import io
from pathlib import Path
from dotenv import load_dotenv

# Force UTF-8 output on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

CHROMA_DIR = Path("data/chroma")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5

SYSTEM_PROMPT = """You are a veterinary data science assistant.
Answer the question using ONLY the context provided below.
If the answer is not in the context, say "I don't have enough information in the loaded papers to answer this."
Always cite which paper (source filename) your answer draws from.

Context:
{context}
"""


def load_retriever():
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    db = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
        collection_name="vetlit",
    )
    count = db._collection.count()
    if count == 0:
        print("ChromaDB is empty -- run ingest.py first.")
        exit(1)
    print(f"  >> {count} vectors loaded from ChromaDB")
    return db.as_retriever(search_kwargs={"k": TOP_K})


def format_docs(docs):
    parts = []
    for doc in docs:
        src = doc.metadata.get("source", "unknown")
        parts.append(f"[{src}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def build_chain(retriever):
    llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ])
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


def show_sources(retriever, question):
    docs = retriever.invoke(question)
    print("\n-- Retrieved chunks ------------------------------------------")
    for i, doc in enumerate(docs, 1):
        src = doc.metadata.get("source", "?")
        preview = doc.page_content[:200].replace("\n", " ")
        print(f"  [{i}] {src}")
        print(f"      {preview}...")
    print("--------------------------------------------------------------\n")


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else None
    if not question:
        question = input("Question: ").strip()

    print(f"\n=== VetLit-RAG Query ===")
    print(f"Q: {question}\n")
    retriever = load_retriever()
    show_sources(retriever, question)
    chain = build_chain(retriever)
    print("-- Answer ----------------------------------------------------")
    print(chain.invoke(question))
    print("--------------------------------------------------------------\n")
