"""Streamlit UI for Codyssey RAG chatbot."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import streamlit_authenticator as stauth
import yaml
from langchain_community.callbacks import get_openai_callback

from app import usage
from app.config import ADMIN_USERS, CHROMA_DIR, EMBEDDING_MODEL, LLM_MODEL
from app.rag import build_chain

print(f"[DEBUG] CHROMA_DIR resolves to: {CHROMA_DIR}", flush=True)
print(f"[DEBUG] CHROMA_DIR exists: {CHROMA_DIR.exists()}", flush=True)
print(f"[DEBUG] CHROMA_DIR contents: {list(CHROMA_DIR.iterdir()) if CHROMA_DIR.exists() else 'N/A'}", flush=True)

RELOAD_INTERVAL = 300  # 5 minutes
AUTH_CONFIG_PATH = Path(__file__).resolve().parent / "auth_config.yaml"


def _load_auth_config() -> dict:
    """Load auth config from Streamlit secrets (cloud) or yaml file (local)."""
    if "auth_config_yaml" in st.secrets:
        return yaml.safe_load(st.secrets["auth_config_yaml"])
    with open(AUTH_CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_authenticator() -> stauth.Authenticate:
    cfg = _load_auth_config()
    return stauth.Authenticate(
        cfg["credentials"],
        cfg["cookie"]["name"],
        cfg["cookie"]["key"],
        cfg["cookie"]["expiry_days"],
    )


def _init_chain() -> None:
    if "chain" not in st.session_state:
        st.session_state.chain, st.session_state.retriever, st.session_state.vs = build_chain()
        st.session_state.last_reload = time.time()
    elif time.time() - st.session_state.last_reload > RELOAD_INTERVAL:
        st.session_state.chain, st.session_state.retriever, st.session_state.vs = build_chain()
        st.session_state.last_reload = time.time()

    if "messages" not in st.session_state:
        st.session_state.messages = []


def _render_admin_sidebar() -> None:
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 사용량")

    tab_month, tab_today = st.sidebar.tabs(["이번 달", "오늘"])

    for tab, since in [(tab_month, usage.month_start_iso()), (tab_today, usage.day_start_iso())]:
        rows = usage.aggregate_by_user(since_iso=since)
        with tab:
            if not rows:
                st.caption("기록 없음")
                continue
            total_q = sum(r["queries"] for r in rows)
            total_cost = sum(r["cost_usd"] for r in rows)
            for r in rows:
                st.write(
                    f"- **{r['username']}** · {r['queries']}회 · ${r['cost_usd']:.4f}"
                )
            st.caption(f"합계: {total_q}회 · ${total_cost:.4f}")


def _render_retrieved_docs(docs: list[dict], dense_scored: list[dict] | None = None) -> None:
    if dense_scored is not None:
        metric = dense_scored[0].get("_metric", "?") if dense_scored else "?"
        with st.expander(f"🧪 Dense top-{len(dense_scored)} with scores (metric: {metric})"):
            if not dense_scored:
                st.warning("Dense 검색이 0건 반환. 컬렉션이 비었거나 임베딩이 실패.")
            for i, d in enumerate(dense_scored, 1):
                section = d.get("section") or "—"
                st.markdown(
                    f"**{i}. score = `{d['score']:.4f}` — {d['title']}**  \n"
                    f"섹션: `{section}`"
                )
                st.code(d["content"][:300] + ("..." if len(d["content"]) > 300 else ""), language="markdown")

    with st.expander(f"🔍 Retrieved chunks ({len(docs)})"):
        for i, d in enumerate(docs, 1):
            section = d.get("section") or "—"
            st.markdown(
                f"**{i}. {d['title']}**  \n"
                f"섹션: `{section}`  \n"
                f"`note_id: {d['note_id']}`"
            )
            st.code(d["content"], language="markdown")


def _diagnose_dense(question: str, k: int = 10) -> list[dict]:
    vs = st.session_state.vs
    metric = (vs._collection.metadata or {}).get("hnsw:space", "l2 (default)")
    pairs = vs.similarity_search_with_relevance_scores(question, k=k)
    return [
        {
            "score": score,
            "title": doc.metadata.get("title", "제목 없음"),
            "section": doc.metadata.get("section", ""),
            "content": doc.page_content,
            "_metric": metric,
        }
        for doc, score in pairs
    ]


def _render_chat(username: str, is_admin: bool) -> None:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if is_admin and msg.get("retrieved_docs"):
                _render_retrieved_docs(
                    msg["retrieved_docs"],
                    dense_scored=msg.get("dense_scored"),
                )

    if question := st.chat_input("질문을 입력하세요"):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("검색 중..."):
                with get_openai_callback() as cb:
                    answer = st.session_state.chain.invoke(question)
                docs = st.session_state.retriever.invoke(question)
                retrieved = [
                    {
                        "title": d.metadata.get("title", "제목 없음"),
                        "note_id": d.metadata.get("note_id", "—"),
                        "section": d.metadata.get("section", ""),
                        "content": d.page_content,
                    }
                    for d in docs
                ]
                dense_scored = _diagnose_dense(question) if is_admin else None
                usage.log(
                    username=username,
                    prompt_tokens=cb.prompt_tokens,
                    completion_tokens=cb.completion_tokens,
                    cost_usd=cb.total_cost,
                )

            st.markdown(answer)
            if is_admin:
                _render_retrieved_docs(retrieved, dense_scored=dense_scored)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "retrieved_docs": retrieved,
            "dense_scored": dense_scored,
        })


def _render_model_footer() -> None:
    st.markdown(
        f"""
        <div style='position: fixed; bottom: 8px; right: 14px;
                    font-size: 11px; color: rgba(128, 128, 128, 0.75);
                    background: rgba(0, 0, 0, 0.04); padding: 4px 10px;
                    border-radius: 6px; z-index: 9999; pointer-events: none;
                    font-family: ui-monospace, SFMono-Regular, monospace;'>
            LLM: {LLM_MODEL} · Embeddings: {EMBEDDING_MODEL}
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="코디세이 AI 도우미", page_icon="🤖")

    authenticator = _load_authenticator()
    authenticator.login(location="main")

    auth_status = st.session_state.get("authentication_status")
    if auth_status is False:
        st.error("아이디 또는 비밀번호가 올바르지 않습니다.")
        st.stop()
    if auth_status is None:
        st.info("로그인이 필요합니다.")
        st.stop()

    st.title("코디세이 퍼실 도우미 챗봇")
    username = st.session_state["username"]
    with st.sidebar:
        st.write(f"👤 {st.session_state['name']}")
        authenticator.logout(location="sidebar")

    is_admin = username in ADMIN_USERS
    if is_admin:
        _render_admin_sidebar()

    _init_chain()
    _render_chat(username, is_admin)
    _render_model_footer()


if __name__ == "__main__":
    main()
