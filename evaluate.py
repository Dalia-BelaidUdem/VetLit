"""
evaluate.py — Measure retrieval and answer quality with RAGAS
Usage: python evaluate.py
"""

from pathlib import Path
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset

load_dotenv()

CHROMA_DIR = Path("data/chroma")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5

TEST_QUESTIONS = [
    "What factors affect productive longevity in dairy cows?",
    "How is mastitis detected early in cattle?",
    "What machine learning methods are used to predict animal disease?",
    "What are the key welfare indicators for dairy cattle?",
    "How does body condition score relate to reproductive performance?",
]

GROUND_TRUTHS = [
    "Productive longevity is influenced by milk production, health events, reproduction, and farm management factors.",
    "Mastitis can be detected via somatic cell count, milk conductivity, and automated milking sensor data.",
    "Random forests, XGBoost, and neural networks are commonly used for disease prediction in livestock.",
    "Key welfare indicators include lameness score, body condition score, somatic cell count, and mortality rate.",
    "Lower body condition score at calving is associated with reduced fertility and longer calving intervals.",
]

SYSTEM_PROMPT = """You are a veterinary data science assistant.
Answer using ONLY the context below. Cite sources.

Context:
{context}
"""

def format_docs(docs):
    return "\n\n---\n\n".join(
        f"[{d.metadata.get('source','?')}]\n{d.page_content}" for d in docs
    )

def run_eval():
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    db = Chroma(persist_directory=str(CHROMA_DIR), embedding_function=embeddings, collection_name="vetlit")
    retriever = db.as_retriever(search_kwargs={"k": TOP_K})
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

    print("Running evaluation questions...\n")
    questions, answers, contexts, ground_truths = [], [], [], []

    for q, gt in zip(TEST_QUESTIONS, GROUND_TRUTHS):
        print(f"  Q: {q[:60]}...")
        docs = retriever.invoke(q)
        answer = chain.invoke(q)
        questions.append(q)
        answers.append(answer)
        contexts.append([d.page_content for d in docs])
        ground_truths.append(gt)

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

    print("\nComputing RAGAS metrics...")
    results = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )

    print("\n── RAGAS Results ─────────────────────────────")
    print(results)
    print("──────────────────────────────────────────────")
    results.to_pandas().to_csv("data/eval_results.csv", index=False)
    print("Saved to data/eval_results.csv\n")

if __name__ == "__main__":
    run_eval()
