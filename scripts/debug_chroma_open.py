"""Open chromadb directly and show the REAL underlying exception."""

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import CHROMA_DIR

print(f"Opening: {CHROMA_DIR}")

import chromadb
from chromadb.config import Settings

settings = Settings(anonymized_telemetry=False, allow_reset=True)

try:
    client = chromadb.PersistentClient(path=str(CHROMA_DIR), settings=settings)
    print(f"Client OK. heartbeat={client.heartbeat()}")
    print(f"Collections: {[c.name for c in client.list_collections()]}")
except Exception:
    print("\n--- REAL TRACEBACK ---")
    traceback.print_exc()
