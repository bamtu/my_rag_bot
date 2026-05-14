"""Inspect chroma.sqlite3 schema to diagnose tenant errors."""

import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import CHROMA_DIR

db_path = CHROMA_DIR / "chroma.sqlite3"
print(f"DB path: {db_path}")
print(f"Exists: {db_path.exists()}")
if not db_path.exists():
    sys.exit(0)
print(f"Size: {db_path.stat().st_size}")

c = sqlite3.connect(str(db_path))
tables = [r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
print(f"Tables: {tables}")

if "tenants" in tables:
    rows = c.execute("SELECT * FROM tenants").fetchall()
    print(f"tenants rows: {rows}")
else:
    print("⚠ no 'tenants' table — this is the old chroma format")

if "databases" in tables:
    cols = [r[1] for r in c.execute("PRAGMA table_info(databases)").fetchall()]
    print(f"databases columns: {cols}")
    rows = c.execute("SELECT * FROM databases").fetchall()
    print(f"databases rows: {rows}")

if "collections" in tables:
    cols = [r[1] for r in c.execute("PRAGMA table_info(collections)").fetchall()]
    print(f"collections columns: {cols}")
    rows = c.execute("SELECT * FROM collections").fetchall()
    print(f"collections rows: {rows}")

print(f"\nmigrations:")
for r in c.execute("SELECT * FROM migrations ORDER BY rowid DESC LIMIT 5"):
    print(f"  {r}")

print(f"\nchromadb version:")
import chromadb
print(f"  {chromadb.__version__}")
