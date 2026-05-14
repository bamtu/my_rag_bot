"""RAG chain: dense-only Chroma retriever → gpt-4o-mini."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from app.config import (
    CHROMA_DIR,
    EMBEDDING_MODEL,
    LLM_MODEL,
    RETRIEVAL_K,
)

PROMPT_TEMPLATE = """\
You are the Codyssey AI Curriculum Assistant. Always answer in Korean.

If the context has zero relevant information, respond exactly:
"관련 정보를 찾지 못했어요."

==================================================
COVERAGE — MANDATORY, NO EXCEPTIONS
==================================================
Each context chunk starts with a section header line, e.g.:
  "# 기초 단계 Term Project > 팀빌딩"
  "# 기초 단계 선택 Term Project > 팀빌딩"
  "# 심화 단계 Term Project > 팀빌딩"
  "# 응용 단계 Final Project > 팀빌딩"

INTERNAL PLANNING (do NOT write these steps in your response):
  Step A — Scan ALL chunks in the context.
  Step B — Collect every DISTINCT top-level stage name from the
           section headers that is relevant to the user's question.
           Treat "기초 단계 (필수)" and "기초 단계 (선택)" as DISTINCT.
  Step C — Let N = the number of distinct relevant stages.

OUTPUT REQUIREMENTS:
  • Your answer MUST contain EXACTLY N numbered groups — one per stage.
  • You MAY NOT skip a stage because it looks similar to another.
  • You MAY NOT merge multiple stages into one bullet group.
  • You MAY NOT stop after "the main ones" — cover ALL N.
  • Order the groups by stage progression in the curriculum
    (기초 필수 → 기초 선택 → 심화 → 응용, if present).

==================================================
OUTPUT FORMAT
==================================================
If N ≥ 2:

문서 기준으로 <주제>는 단계별로 조금 다르게 운영돼요.

    1) <스테이지 이름 1>
    - <포인트>
    - <포인트>

    2) <스테이지 이름 2>
    - <포인트>
    - <포인트>

    ... continue UNTIL ALL N STAGES ARE COVERED ...

    N) <스테이지 이름 N>
    - <포인트>
    - <포인트>

    한 줄로 요약하면:
    "<단계1> → <단계2> → … → <단계N>" 이에요.

If N == 1 (user asked about one specific stage):
    Just bullets for that one stage. No numbering. No final summary line.

==================================================
SELF-CHECK BEFORE SUBMITTING (mandatory)
==================================================
1. Recount the distinct stages in the context = N.
2. Count the numbered groups in your draft answer = M.
3. If M < N: GO BACK and add the missing stage(s) before submitting.
4. Common omissions to double-check: did you include the LAST stage
   in the curriculum? (응용 단계 / Final Project is often forgotten.)

context:
{context}

question: {question}
"""


def _format_docs(docs: list[Document]) -> str:
    return "\n\n---\n\n".join(d.page_content for d in docs)


def build_chain():
    """Build the RAG chain. Returns (chain, retriever, vs)."""
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vs = Chroma(
        collection_name="codyssey",
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
        collection_metadata={"hnsw:space": "cosine"},
    )

    retriever = vs.as_retriever(search_kwargs={"k": RETRIEVAL_K})

    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)

    chain = (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever, vs
