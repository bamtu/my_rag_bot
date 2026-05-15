import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(override=True)

# Joplin (인덱서에서만 사용. Streamlit 런타임에는 불필요해서 optional)
JOPLIN_TOKEN = os.getenv("JOPLIN_TOKEN")
# 설정 시 해당 노트(들)만 인덱싱. 콤마 구분으로 여러 ID 가능. 미설정이면 전체 노트.
_raw_ids = os.getenv("TARGET_NOTE_IDS") or os.getenv("TARGET_NOTE_ID") or ""
TARGET_NOTE_IDS: list[str] | None = (
    [s.strip() for s in _raw_ids.split(",") if s.strip()] or None
)

# OpenAI
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
LLM_MODEL = os.environ["LLM_MODEL"]

# Embeddings — provider: "openai" or "huggingface"
EMBEDDING_PROVIDER = os.environ["EMBEDDING_PROVIDER"]
EMBEDDING_MODEL = os.environ["EMBEDDING_MODEL"]

# Retrieval (dense-only, plain top-k similarity)
RETRIEVAL_K = int(os.environ["RETRIEVAL_K"])

# Admin usernames (comma-separated) — see app/auth_config.yaml for usernames
ADMIN_USERS = {u.strip() for u in os.environ["ADMIN_USERS"].split(",") if u.strip()}

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
