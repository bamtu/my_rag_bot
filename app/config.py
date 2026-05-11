import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Joplin
JOPLIN_TOKEN = os.environ["JOPLIN_TOKEN"]
TARGET_NOTE_ID = os.getenv("TARGET_NOTE_ID")  # 설정 시 해당 노트만 인덱싱/참조

# OpenAI
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = DATA_DIR / "chroma"
INDEX_STATE_PATH = DATA_DIR / "index_state.json"
BM25_DOCS_PATH = DATA_DIR / "bm25_docs.pkl"

# Ensure data directories exist
DATA_DIR.mkdir(exist_ok=True)
CHROMA_DIR.mkdir(exist_ok=True)
