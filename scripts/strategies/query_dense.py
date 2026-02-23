"""Run a single Dense strategy query with real corpus and LLM."""
import os
import argparse
from dotenv import load_dotenv
load_dotenv()

output_dir = os.getenv("OUTPUT_DIR", "data/processed")
corpus_path = f"{output_dir}/corpus.jsonl"
index_path = f"{output_dir}/vector_index.faiss"
meta_path = f"{output_dir}/vector_meta.parquet"

from raqr.strategies.dense import DenseStrategy
from raqr.index_store import FaissIndexStore
from raqr.loaders import VectorMetaMapper, JsonCorpusLoader
from raqr.embedder import SentenceTransformersEmbedder
from raqr.generator import SimpleLLMGenerator

strategy = DenseStrategy(
    index_store=FaissIndexStore(index_path=index_path),
    meta=VectorMetaMapper(parquet_path=meta_path),
    embedder=SentenceTransformersEmbedder(model_name="all-MiniLM-L6-v2"),
    generator=SimpleLLMGenerator(
        model_id=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        base_prompt="Answer the question based only on the provided context. If the context does not contain the answer, say so.",
    ),
    corpus=JsonCorpusLoader(jsonl_path=corpus_path),
    top_k=int(os.getenv("DENSE_TOP_K", 10)),
)

# Parse argument
parser = argparse.ArgumentParser(description="A simple RAG testing script")
parser.add_argument("query", help="The question you want to ask the LLM")
args = parser.parse_args()

# Your question here
question = args.query
result = strategy.retrieve_and_generate(question)

print("Status:", result.status)
if result.status == "ERROR":
    print(result.error)
print("Answer:", result.answer)
print("\n\n---\n\n")
print("Latency (ms):", result.latency_ms)
print("\n\n---\n\n")

for i in range(5):
    print(f"Context {i + 1}:", result.context_scores[i][0])
    print("\n\n---\n\n")

# Run: poetry run python scripts/strategies/query_dense.py "<Question>"