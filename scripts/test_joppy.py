"""Joplin 연결 및 노트 수신 확인용 진단 스크립트.

실행: PYTHONPATH=. python scripts/test_joppy.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.joplin_client import fetch_all_notes, get_api


def main():
    print("[1] Joplin ClientApi 연결 시도 (token 기반, localhost:41184)")
    api = get_api()
    print("    -> ClientApi 객체 생성 OK")

    print("\n[2] 전체 노트 가져오기...")
    notes = fetch_all_notes(api)
    print(f"    -> {len(notes)}개 노트 수신")

    if not notes:
        print("\n경고: 노트가 0개입니다. Joplin Server에 노트가 있는지 확인하세요.")
        return

    print(f"\n[3] 전체 {len(notes)}개 노트 목록:")
    for i, n in enumerate(notes, 1):
        body_preview = (n["body"] or "")[:120].replace("\n", " ")
        print(f"\n  [{i}] id={n['id']}")
        print(f"      title: {n['title']}")
        print(f"      updated_time: {n['updated_time']}")
        print(f"      body length: {len(n['body'] or '')}자")
        print(f"      body preview: {body_preview}")

    print("\n[4] 통계:")
    total_chars = sum(len(n["body"] or "") for n in notes)
    empty_notes = sum(1 for n in notes if not (n["body"] or "").strip())
    print(f"    - 전체 본문 글자수: {total_chars:,}")
    print(f"    - 빈 노트 개수: {empty_notes}")
    print(f"    - 평균 노트 길이: {total_chars // len(notes):,}자")

    print("\n성공: joppy로 Joplin 노트를 정상 수신했습니다.")


if __name__ == "__main__":
    main()
