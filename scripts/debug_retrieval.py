"""Diagnose retrieval: check collection distance metric and score distribution."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from app.config import CHROMA_DIR, EMBEDDING_MODEL

QUERY = sys.argv[1] if len(sys.argv) > 1 else "팀빌딩"
TOP_N = 15

embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
vs = Chroma(
    collection_name="codyssey",
    embedding_function=embeddings,
    persist_directory=str(CHROMA_DIR),
)

raw = vs._collection.metadata or {}
print(f"Collection metadata: {raw}")
print(f"Distance metric: {raw.get('hnsw:space', 'l2 (default)')}")
print(f"Total chunks: {vs._collection.count()}")
print()
print(f"Query: {QUERY!r}")
print(f"Top {TOP_N} results by relevance score:")
print("-" * 80)

results = vs.similarity_search_with_relevance_scores(QUERY, k=TOP_N)
for i, (doc, score) in enumerate(results, 1):
    title = doc.metadata.get("title", "?")
    preview = doc.page_content[:80].replace("\n", " ")
    print(f"{i:2d}. score={score:.4f}  {title}")
    print(f"    {preview}...")
