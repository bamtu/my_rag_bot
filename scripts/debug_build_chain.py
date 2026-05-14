"""Run build_chain() exactly as Streamlit does, surface real traceback."""

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import CHROMA_DIR
print(f"CHROMA_DIR resolves to: {CHROMA_DIR}")
print(f"Exists: {CHROMA_DIR.exists()}")
print(f"Contents: {list(CHROMA_DIR.iterdir())}")

try:
    from app.rag import build_chain
    chain, retriever = build_chain()
    print("\nbuild_chain() OK")
    print(f"Retriever: {type(retriever).__name__}")
except Exception:
    print("\n--- REAL TRACEBACK ---")
    traceback.print_exc()
