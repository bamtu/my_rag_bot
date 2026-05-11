"""RAG chain: EnsembleRetriever (dense + BM25) → gpt-4o-mini."""

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_classic.retrievers import EnsembleRetriever
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from app.config import BM25_DOCS_PATH, CHROMA_DIR, EMBEDDING_MODEL, LLM_MODEL
from app.tokenizer import korean_tokenize

PROMPT_TEMPLATE = """\
당신은 코디세이 AI 올인원 교육과정 도우미입니다.
아래 컨텍스트를 참고하여 질문에 한국어로 답변하세요.
컨텍스트에 없는 내용은 "관련 자료를 찾지 못했습니다"라고 답하세요.

컨텍스트:
{context}

질문: {question}
"""


def _load_bm25_docs() -> list[Document]:
    path = Path(BM25_DOCS_PATH)
    if not path.exists():
        return []
    with open(path, "rb") as f:
        return pickle.load(f)


def _format_docs(docs: list[Document]) -> str:
    return "\n\n---\n\n".join(d.page_content for d in docs)


def build_chain():
    """Build the RAG chain. Returns (chain, retriever)."""
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vs = Chroma(
        collection_name="codyssey",
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )
    dense_retriever = vs.as_retriever(search_kwargs={"k": 10})

    bm25_docs = _load_bm25_docs()

    if bm25_docs:
        bm25_retriever = BM25Retriever.from_documents(
            bm25_docs,
            preprocess_func=korean_tokenize,
            k=10,
        )
        retriever = EnsembleRetriever(
            retrievers=[dense_retriever, bm25_retriever],
            weights=[0.6, 0.4],
        )
    else:
        retriever = dense_retriever

    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)

    chain = (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever
