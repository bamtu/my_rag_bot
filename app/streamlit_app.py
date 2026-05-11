"""Streamlit UI for Codyssey RAG chatbot."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from app.rag import build_chain

RELOAD_INTERVAL = 300  # 5 minutes


def _init():
    if "chain" not in st.session_state:
        st.session_state.chain, st.session_state.retriever = build_chain()
        st.session_state.last_reload = time.time()
    elif time.time() - st.session_state.last_reload > RELOAD_INTERVAL:
        st.session_state.chain, st.session_state.retriever = build_chain()
        st.session_state.last_reload = time.time()

    if "messages" not in st.session_state:
        st.session_state.messages = []


def main():
    st.set_page_config(page_title="코디세이 AI 도우미", page_icon="🤖")
    st.title("코디세이 AI 교육과정 도우미")

    _init()

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📄 출처"):
                    for src in msg["sources"]:
                        st.write(f"- {src}")

    # Chat input
    if question := st.chat_input("질문을 입력하세요"):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("검색 중..."):
                # Get answer
                answer = st.session_state.chain.invoke(question)

                # Get sources
                docs = st.session_state.retriever.invoke(question)
                sources = list(dict.fromkeys(
                    d.metadata.get("title", "제목 없음") for d in docs[:5]
                ))

            st.markdown(answer)
            if sources:
                with st.expander("📄 출처"):
                    for src in sources:
                        st.write(f"- {src}")

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
        })


if __name__ == "__main__":
    main()
