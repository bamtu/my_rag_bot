# Codyssey RAG Bot

Joplin Server 노트를 RAG(Retrieval-Augmented Generation)로 검색·답변하는 한국어 챗봇.

- **Dense + Sparse 하이브리드 검색**: ChromaDB(임베딩) + BM25(Kiwi 한국어 토크나이저)
- **증분 인덱싱**: Joplin 노트의 `updated_time`을 추적해 변경분만 재임베딩
- **Streamlit 채팅 UI**: 답변과 함께 출처 노트 제목 표시

## 요구 사항

- Python 3.10+
- 실행 중인 Joplin Server (또는 Web Clipper API 활성화)
- OpenAI API 키

## 설치

```bash
python3 -m venv venv
source venv/bin/activate           # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 설정

`.env.example`을 복사하여 `.env`를 만들고 값을 채웁니다.

```ini
JOPLIN_TOKEN=your_joplin_webclipper_token
OPENAI_API_KEY=sk-...

# 선택값 (기본값 사용 시 생략 가능)
LLM_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small

# 특정 노트 하나만 인덱싱하려면 ID 지정
# TARGET_NOTE_ID=77a9ba0ee62742ca96fb5f27e245d70b
```

| 변수 | 설명 | 기본값 |
|---|---|---|
| `JOPLIN_TOKEN` | Joplin Web Clipper API 토큰 | (필수) |
| `OPENAI_API_KEY` | OpenAI API 키 | (필수) |
| `LLM_MODEL` | 답변 생성 모델 | `gpt-4o-mini` |
| `EMBEDDING_MODEL` | 임베딩 모델 | `text-embedding-3-small` |
| `TARGET_NOTE_ID` | 단일 노트만 인덱싱/참조할 때 지정 | (전체) |

## 실행

### 1) 인덱싱

최초 1회 실행하면 모든 노트가 임베딩되고, 이후 실행 시 변경/삭제된 노트만 갱신됩니다.

```bash
PYTHONPATH=. python -m app.indexer
```

### 2) 챗봇 UI

```bash
streamlit run app/streamlit_app.py --server.address 0.0.0.0 --server.port 8501
```

브라우저에서 `http://localhost:8501` 접속.

> UI는 5분(`RELOAD_INTERVAL`)마다 RAG 체인을 자동 재로드하여 최신 인덱스를 반영합니다.

## 아키텍처

```
Joplin Server ──► Indexer ──► ChromaDB (dense)
                       └────► bm25_docs.pkl (sparse corpus)
                                │
                                ▼
                        EnsembleRetriever (0.6 dense + 0.4 BM25)
                                │
                                ▼
                          ChatOpenAI (gpt-4o-mini)
                                │
                                ▼
                          Streamlit UI
```

### 인덱싱 파이프라인 (`app/indexer.py`)

1. Joplin에서 노트를 가져와 `index_state.json`의 `updated_time`과 비교
2. 변경된 노트: `MarkdownHeaderTextSplitter`로 헤더 단위 분할 → 1500자 초과 시 `RecursiveCharacterTextSplitter`로 재분할 (chunk_size 1000, overlap 150)
3. 변경/삭제 노트의 기존 청크를 ChromaDB에서 제거 후 재추가
4. BM25용으로 전체 청크를 `bm25_docs.pkl`에 덤프

### 검색 & 답변 (`app/rag.py`)

- `Chroma` dense retriever (`k=10`) + `BM25Retriever` (`k=10`, Kiwi 형태소 분석)
- `EnsembleRetriever` 가중치 **dense 0.6 / BM25 0.4**
- 컨텍스트에 없는 내용은 "관련 자료를 찾지 못했습니다"로 응답하도록 프롬프트 제약

### 한국어 토크나이저 (`app/tokenizer.py`)

Kiwi로 형태소 분석 후 명사(N), 동사(V), 외국어(SL), 숫자(SN) 태그만 BM25 토큰으로 사용.

## 디렉토리 구조

```
my_rag_bot/
├── app/
│   ├── config.py           # 환경변수·경로 로드
│   ├── joplin_client.py    # Joplin API 래퍼 (joppy)
│   ├── indexer.py          # 증분 인덱서
│   ├── rag.py              # EnsembleRetriever + LLM 체인
│   ├── tokenizer.py        # Kiwi 한국어 토크나이저
│   └── streamlit_app.py    # 챗봇 UI
├── data/                   # 인덱싱 결과 (gitignore)
│   ├── chroma/             # ChromaDB persist
│   ├── index_state.json    # 노트별 updated_time 스냅샷
│   └── bm25_docs.pkl       # BM25 corpus 덤프
├── scripts/
│   └── test_joppy.py       # Joplin 연결 테스트
├── .env.example
├── requirements.txt
└── README.md
```

## 운영 팁

- **노트를 추가/수정한 뒤** `python -m app.indexer`를 다시 실행하면 변경분만 반영됩니다.
- **인덱스를 완전히 초기화**하려면 `data/` 디렉토리를 삭제하고 인덱서를 다시 실행하세요.
- **검색 품질 튜닝**은 `app/rag.py`의 EnsembleRetriever `weights`와 retriever `k` 값을 조정합니다.
