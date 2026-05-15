"""Incremental indexer: fetches changed notes from Joplin, re-embeds, updates BM25 dump."""

import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from app.config import (
    BM25_DOCS_PATH,
    CHROMA_DIR,
    INDEX_STATE_PATH,
    TARGET_NOTE_IDS,
)
from app.embeddings import get_embeddings
from app.joplin_client import fetch_notes, get_api

HEADERS_TO_SPLIT = [
    ("#", "h1"),
    ("##", "h2"),
]

md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=HEADERS_TO_SPLIT)
recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
)


def _header_path(meta: dict) -> str:
    parts = [meta[k] for k in ("h1", "h2") if meta.get(k)]
    return " > ".join(parts)


def chunk_note(note: dict) -> list[Document]:
    """Split a note by markdown headers; prepend header path to each chunk so
    embedding and BM25 see the section context (e.g. '1단계 > 팀빌딩')."""
    body = note["body"] or ""
    if not body.strip():
        return []

    md_chunks = md_splitter.split_text(body)
    if not md_chunks:
        md_chunks = [Document(page_content=body)]

    final_chunks = []
    for chunk in md_chunks:
        text = chunk.page_content if isinstance(chunk, Document) else chunk
        meta = dict(chunk.metadata) if isinstance(chunk, Document) and chunk.metadata else {}
        section = _header_path(meta)

        sub_texts = recursive_splitter.split_text(text) if len(text) > 1500 else [text]

        for sub in sub_texts:
            content = f"# {section}\n\n{sub}" if section else sub
            final_chunks.append(
                Document(
                    page_content=content,
                    metadata={
                        **meta,
                        "note_id": note["id"],
                        "title": note["title"],
                        "section": section,
                    },
                )
            )

    return final_chunks


def load_state() -> dict:
    if INDEX_STATE_PATH.exists():
        return json.loads(INDEX_STATE_PATH.read_text(encoding="utf-8"))
    return {}


def save_state(state: dict) -> None:
    INDEX_STATE_PATH.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")


def dump_bm25_corpus(vs: Chroma) -> None:
    """Dump all documents from ChromaDB for BM25 retriever."""
    result = vs.get(include=["documents", "metadatas"])
    docs = []
    for text, meta in zip(result["documents"], result["metadatas"]):
        docs.append(Document(page_content=text, metadata=meta))
    with open(BM25_DOCS_PATH, "wb") as f:
        pickle.dump(docs, f)
    print(f"BM25 corpus dumped: {len(docs)} chunks")


def run() -> None:
    print("Connecting to Joplin...")
    api = get_api()
    if TARGET_NOTE_IDS:
        print(f"TARGET_NOTE_IDS 설정됨 -> {len(TARGET_NOTE_IDS)}개 노트만 인덱싱: {TARGET_NOTE_IDS}")
    notes = fetch_notes(api, target_note_ids=TARGET_NOTE_IDS)
    print(f"Fetched {len(notes)} notes from Joplin")

    old_state = load_state()
    current_ids = {n["id"] for n in notes}

    # Detect changes
    to_update = [
        n for n in notes if str(n["updated_time"]) != str(old_state.get(n["id"]))
    ]
    to_delete = [nid for nid in old_state if nid not in current_ids]

    if not to_update and not to_delete:
        print("No changes detected.")
        return

    print(f"Updating {len(to_update)} notes, deleting {len(to_delete)} notes")

    embeddings = get_embeddings()
    vs = Chroma(
        collection_name="codyssey",
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
        collection_metadata={"hnsw:space": "cosine"},
    )

    # Delete removed notes
    for nid in to_delete:
        vs.delete(where={"note_id": nid})
        print(f"  Deleted chunks for note {nid}")

    # Re-index changed notes
    for note in to_update:
        # Remove old chunks first
        vs.delete(where={"note_id": note["id"]})
        chunks = chunk_note(note)
        if chunks:
            ids = [f"{note['id']}_{i}" for i in range(len(chunks))]
            vs.add_documents(documents=chunks, ids=ids)
        print(f"  Indexed '{note['title']}': {len(chunks)} chunks")

    # Build new state
    new_state = {n["id"]: str(n["updated_time"]) for n in notes}
    save_state(new_state)

    # Dump BM25 corpus
    dump_bm25_corpus(vs)

    print("Indexing complete.")


if __name__ == "__main__":
    run()
