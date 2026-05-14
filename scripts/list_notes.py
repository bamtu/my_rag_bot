"""List all Joplin notes with their IDs and titles.

Usage:
  python scripts\list_notes.py            # list all
  python scripts\list_notes.py 팀빌딩     # filter by keyword in title
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.joplin_client import fetch_all_notes, get_api

keyword = sys.argv[1] if len(sys.argv) > 1 else None

api = get_api()
notes = fetch_all_notes(api)

if keyword:
    notes = [n for n in notes if keyword.lower() in (n["title"] or "").lower()]

print(f"{len(notes)} note(s) found\n")
print(f"{'ID':<35}  {'TITLE'}")
print(f"{'-' * 35}  {'-' * 50}")
for n in notes:
    title = (n["title"] or "").strip() or "(제목 없음)"
    print(f"{n['id']:<35}  {title}")
