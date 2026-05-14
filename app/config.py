import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Joplin
JOPLIN_TOKEN = os.environ["JOPLIN_TOKEN"]
# 설정 시 해당 노트(들)만 인덱싱. 콤마 구분으로 여러 ID 가능. 미설정이면 전체 노트.
_raw_ids = os.getenv("TARGET_NOTE_IDS") or os.getenv("TARGET_NOTE_ID") or ""
TARGET_NOTE_IDS: list[str] | None = (
    [s.strip() for s in _raw_ids.split(",") if s.strip()] or None
)

# OpenAI
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# Retrieval (dense-only, plain top-k)
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "10"))

# Admin usernames (comma-separated) — see app/auth_config.yaml for usernames
ADMIN_USERS = {u.strip() for u in os.getenv("ADMIN_USERS", "admin").split(",") if u.strip()}

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = DATA_DIR / "chroma"
INDEX_STATE_PATH = DATA_DIR / "index_state.json"
BM25_DOCS_PATH = DATA_DIR / "bm25_docs.pkl"
USAGE_DB_PATH = DATA_DIR / "usage.db"

# Ensure data directories exist
DATA_DIR.mkdir(exist_ok=True)
CHROMA_DIR.mkdir(exist_ok=True)
