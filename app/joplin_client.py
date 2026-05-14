import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from joppy.client_api import ClientApi

from app.config import JOPLIN_TOKEN


def get_api() -> ClientApi:
    if not JOPLIN_TOKEN:
        raise RuntimeError(
            "JOPLIN_TOKEN 미설정. .env 또는 환경변수에 토큰을 넣어야 인덱싱 가능."
        )
    return ClientApi(token=JOPLIN_TOKEN)


FIELDS = "id,title,body,parent_id,updated_time"


def _to_dict(n) -> dict:
    return {
        "id": n.id,
        "title": n.title,
        "body": n.body,
        "updated_time": n.updated_time,
        "parent_id": n.parent_id,
    }


def fetch_all_notes(api: ClientApi) -> list[dict]:
    """Fetch all notes with required fields."""
    notes = api.get_all_notes(fields=FIELDS)
    return [_to_dict(n) for n in notes]


def fetch_note(api: ClientApi, note_id: str) -> dict:
    """Fetch a single note by ID."""
    n = api.get_note(id_=note_id, fields=FIELDS)
    return _to_dict(n)


def fetch_notes(
    api: ClientApi, target_note_ids: list[str] | None = None
) -> list[dict]:
    """Fetch notes - either all notes or the specified subset by ID."""
    if target_note_ids:
        return [fetch_note(api, nid) for nid in target_note_ids]
    return fetch_all_notes(api)
