# Codyssey RAG Bot — 관리자 가이드

Joplin 노트를 RAG(Retrieval-Augmented Generation)로 검색·답변하는 한국어 챗봇.
이 문서는 **관리자(운영자)** 가 앱을 설치·운영·유지보수하기 위한 모든 절차를 담는다.

- **Dense 검색**: ChromaDB(cosine 거리) + OpenAI 임베딩
- **헤더 인지 청킹**: 마크다운 `##`/`###` 구조를 분석해 청크 앞에 섹션 경로(`# 대주제 > 소주제`)를 prepend
- **증분 인덱싱**: `updated_time` 비교로 변경분만 재임베딩
- **Streamlit 챗봇 UI**: 다중 사용자 로그인, 관리자/일반 사용자 권한 분리
- **사용량 추적**: 사용자별 쿼리 수·토큰·비용을 SQLite에 기록 (관리자 전용 사이드바)

---

## 1. 사전 요구 사항

| 항목 | 비고 |
| --- | --- |
| Python 3.10+ | |
| Joplin Desktop | Web Clipper 서비스 활성화 필요 |
| OpenAI API 키 | 임베딩 + LLM 호출 모두 OpenAI 사용 |

---

## 2. 최초 설치

### 2.1 저장소 클론 & 가상환경

```powershell
git clone <repo-url> my_rag_bot
cd my_rag_bot
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

(macOS/Linux는 `source venv/bin/activate`)

### 2.2 환경변수 설정 (`.env`)

`.env.example`을 복사해서 `.env` 생성:

```powershell
copy .env.example .env
```

필수 값을 채운다:

```ini
JOPLIN_TOKEN=your_joplin_webclipper_token
OPENAI_API_KEY=sk-...

# 선택값 — 기본값 사용 시 생략 가능
LLM_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small

# 인덱싱할 노트 ID (콤마 구분, 미설정 시 Joplin 전체 노트)
TARGET_NOTE_IDS=77a9ba0ee62742ca96fb5f27e245d70b,abc1234...

# Retrieval — 한 번에 가져올 청크 수
RETRIEVAL_K=10

# 관리자 사용자명 (auth_config.yaml의 username, 콤마 구분)
ADMIN_USERS=admin
```

**Joplin 토큰 얻는 법**: Joplin Desktop → Tools → Options → Web Clipper → "Enable" 체크 후 표시되는 Authorisation token 복사.

### 2.3 사용자 인증 설정 (`app/auth_config.yaml`)

다중 사용자 로그인 설정 파일을 생성:

```powershell
copy app\auth_config.yaml.example app\auth_config.yaml
```

파일 구조:

```yaml
credentials:
  usernames:
    admin:                                     # 로그인 ID
      name: 관리자                              # 표시 이름
      password: $2b$12$...                     # bcrypt 해시 (아래 절차로 생성)
    alice:
      name: Alice
      password: $2b$12$...

cookie:
  name: codyssey_auth
  key: <랜덤 32바이트 hex>                      # 아래 절차로 생성
  expiry_days: 30
```

**비밀번호 해시·쿠키 키 발급**:

```powershell
python scripts\hash_password.py
```

대화형으로 비밀번호 입력 → bcrypt 해시 + 랜덤 hex 쿠키 키 출력 → 각각 yaml에 붙여 넣기.

> 💡 사용자가 많아 일괄 처리하고 싶으면, yaml에 평문 비밀번호를 그대로 적은 뒤
> `python scripts\hash_yaml_passwords.py`를 실행하면 평문이 자동으로 해시로 교체된다.

---

## 3. Joplin 노트 준비

### 3.1 노트 ID 찾기

Joplin Desktop이 켜져 있고 Web Clipper가 활성화된 상태에서:

```powershell
# 전체 노트 목록 (ID + 제목)
python scripts\list_notes.py

# 제목 키워드로 필터
python scripts\list_notes.py 커리큘럼
```

복사한 ID를 `.env`의 `TARGET_NOTE_IDS`에 붙여 넣는다 (콤마 구분).

### 3.2 노트 작성 가이드 (검색 품질용)

RAG 품질은 마크다운 구조에 크게 의존한다:

- `## 대주제`, `### 소주제` 식으로 계층 구조를 유지
- 동일 토픽(예: "팀빌딩")이 여러 단계에 나오면 각각 별도 `###` 섹션으로 분리
- 청크 분할이 헤더 단위로 일어나므로, 헤더 없이 긴 문단이 이어지면 검색 정확도가 떨어짐

---

## 4. 인덱싱

### 4.1 최초 인덱싱

```powershell
python -m app.indexer
```

- `.env`의 `TARGET_NOTE_IDS`에 해당하는 노트만 가져옴
- 마크다운 헤더 단위로 분할 → 1500자 초과 시 1000자/150 overlap으로 재분할
- 각 청크 앞에 `# 대주제 > 소주제` 경로를 prepend (검색 의미 강화)
- OpenAI 임베딩 호출 → ChromaDB(cosine)에 저장
- 상태 파일 `data/index_state.json`에 노트별 `updated_time` 기록

### 4.2 증분 인덱싱 (노트 추가·수정 후)

```powershell
python -m app.indexer
```

`updated_time`이 바뀐 노트만 재임베딩. 삭제된 노트의 청크는 자동 제거.

### 4.3 노트 추가하기

1. `.env`의 `TARGET_NOTE_IDS`에 새 ID 추가 (`TARGET_NOTE_IDS=기존ID,새로운ID`)
2. 인덱서 재실행: `python -m app.indexer`

### 4.4 인덱스 전체 초기화

청킹 로직이나 임베딩 모델을 바꾼 경우 전체 재구축 필요:

```powershell
Remove-Item -Recurse -Force data\chroma
Remove-Item -Force data\index_state.json
Remove-Item -Force data\bm25_docs.pkl
python -m app.indexer
```

> ⚠️ 전체 노트를 다시 임베딩하므로 OpenAI 비용 발생. `text-embedding-3-small`은 매우 저렴(~$0.02/1M tokens)하지만 노트 양이 많으면 확인 후 진행.

---

## 5. 앱 실행

```powershell
streamlit run app/streamlit_app.py --server.address 0.0.0.0 --server.port 8501
```

브라우저에서 `http://localhost:8501` 접속 → `auth_config.yaml`의 ID/비밀번호로 로그인.

> UI는 5분(`RELOAD_INTERVAL`)마다 RAG 체인을 자동 재로드해 새 인덱스를 반영.
> 코드 변경 시에는 Streamlit 서버를 완전히 재시작해야 적용된다 (Ctrl+C 후 재실행).

---

## 6. 사용자 관리

### 6.1 새 사용자 추가

1. `app/auth_config.yaml`의 `credentials.usernames` 아래에 항목을 추가하고 **비밀번호는 평문 그대로** 적는다:

   ```yaml
   credentials:
     usernames:
       newuser:
         name: 새 사용자
         password: mypassword123       # 평문 그대로
   ```

2. 해시 변환 스크립트 실행:

   ```powershell
   python scripts\hash_yaml_passwords.py
   ```

3. `auth_config.yaml`을 다시 열어 해당 사용자의 `password`가 `$2b$12$...`로 시작하는 bcrypt 해시로 바뀌었는지 확인.
4. Streamlit 서버 재시작.

> 💡 `hash_yaml_passwords.py`는 이미 해시된 항목(`$2`로 시작)은 건드리지 않고, 새로 추가된 평문만 해시한다 — 기존 사용자의 해시는 안전.
>
> 대안으로 `python scripts\hash_password.py`를 쓰면 대화형으로 해시 하나만 받아 직접 yaml에 붙여 넣을 수도 있다.

### 6.2 사용자 삭제

`auth_config.yaml`에서 해당 username 항목을 통째로 제거 → 서버 재시작.

### 6.3 비밀번호 재설정

해당 사용자의 `password` 줄을 새 해시로 교체 → 서버 재시작.

### 6.4 관리자 권한 부여

`.env`의 `ADMIN_USERS`에 username을 콤마로 추가:

```
ADMIN_USERS=admin,alice
```

관리자에게는 두 가지 추가 기능이 보인다:

- **사이드바 사용량 패널**: 사용자별 쿼리 수·비용 (이번 달/오늘 탭)
- **답변 하단 진단 정보**:
  - 🧪 Dense top-K with scores — cosine 유사도 점수와 함께 dense retrieval 결과
  - 🔍 Retrieved chunks — LLM에 실제로 전달된 청크 전문

---

## 7. 검색 품질 튜닝

### 7.1 회수 개수 조정

```ini
RETRIEVAL_K=10
```

- 답변이 일부 토픽만 다루면 → 15~20으로 올림
- 답변에 노이즈가 섞이면 → 6~8로 줄임

### 7.2 LLM 모델 변경

```ini
LLM_MODEL=gpt-4o
```

instruction-following이 부족해서 일부 단계가 답변에서 누락된다면 더 큰 모델로. 비용은 약 10~20배.

### 7.3 임베딩 모델 변경

```ini
EMBEDDING_MODEL=text-embedding-3-large
```

임베딩 모델을 바꾸면 **반드시 인덱스 전체 초기화** (4.4 참고) 후 재인덱싱 필요.

### 7.4 진단 도구

```powershell
# 쿼리에 대한 dense 검색 점수 분포 확인
python scripts\debug_retrieval.py 팀빌딩
```

출력에서 확인할 것:

- `Distance metric:` — `cosine` 이어야 정상 (`l2 (default)`면 재인덱싱 필요)
- 상위 score 분포 — 0.3~0.5 이상이 검색에 잡혀야 의미 매칭 정상

---

## 8. 사용량 모니터링

관리자 사이드바에서 사용자별 쿼리/비용을 실시간 확인. 데이터는 `data/usage.db` (SQLite)에 누적된다.

직접 SQL로 분석하려면:

```powershell
sqlite3 data\usage.db "SELECT username, COUNT(*), SUM(cost_usd) FROM usage GROUP BY username;"
```

---

## 9. 디렉토리 구조

```
my_rag_bot/
├── app/
│   ├── config.py              # 환경변수·경로 로드
│   ├── joplin_client.py       # Joplin API 래퍼
│   ├── indexer.py             # 증분 인덱서 (헤더 prefix 청킹)
│   ├── rag.py                 # Chroma dense retriever + LLM 체인 + 프롬프트
│   ├── tokenizer.py           # Kiwi 한국어 토크나이저 (legacy, 미사용)
│   ├── usage.py               # SQLite 사용량 로깅
│   ├── streamlit_app.py       # 챗봇 UI + 인증 + admin 패널
│   ├── auth_config.yaml.example
│   └── auth_config.yaml       # (생성 후 gitignore)
├── data/                      # 인덱싱 결과 (gitignore)
│   ├── chroma/                # ChromaDB persist (cosine collection)
│   ├── index_state.json       # 노트별 updated_time 스냅샷
│   ├── bm25_docs.pkl          # legacy BM25 corpus (현재 미사용)
│   └── usage.db               # SQLite 사용량 DB
├── scripts/
│   ├── hash_password.py       # bcrypt 해시 + 쿠키 키 생성 (대화형)
│   ├── hash_yaml_passwords.py # auth_config.yaml의 평문 비밀번호 일괄 해시
│   ├── list_notes.py          # Joplin 노트 ID 조회
│   ├── debug_retrieval.py     # dense 검색 점수 진단
│   ├── debug_build_chain.py
│   ├── debug_chroma_open.py
│   └── inspect_chroma.py
├── .env.example
├── requirements.txt
└── README.md
```

---

## 10. 트러블슈팅

| 증상 | 원인 / 조치 |
| --- | --- |
| `requests.ConnectionError: ... port=41184: refused` | Joplin Desktop이 꺼져 있거나 Web Clipper 미활성 → Tools → Options → Web Clipper에서 Enable |
| `No relevant docs were retrieved using the relevance score threshold X.X` | 사용 안 함 (현재 threshold 로직 제거됨). 만약 보인다면 환경변수가 캐시된 상태 — Streamlit 완전 재시작 |
| 답변에 일부 토픽만 등장 | (1) `RETRIEVAL_K`를 늘려 모든 섹션이 회수되는지 admin의 "Retrieved chunks"에서 확인 (2) LLM 모델 업그레이드 (3) 프롬프트의 self-check 규칙 강화 |
| 답변이 엉뚱한 노트 내용 | dense 점수가 낮을 때 발생 — `python scripts\debug_retrieval.py <키워드>`로 진단. metric이 `l2`면 인덱스 전체 초기화 후 재인덱싱 (4.4) |
| 인덱서가 `KeyError: 'id'` 또는 노트 못 찾음 | `TARGET_NOTE_IDS`에 잘못된 ID 또는 placeholder가 남아있음 → `python scripts\list_notes.py`로 실제 ID 확인 |
| 코드 변경 후 동작 안 바뀜 | Streamlit이 session_state에 RAG 체인을 5분 캐싱 → Ctrl+C로 완전 종료 후 재실행 |
| 사용자 추가했는데 로그인 안 됨 | `auth_config.yaml`에 평문이 들어가 있을 수 있음 → `python scripts\hash_yaml_passwords.py` 실행 |

---

## 11. 환경변수 레퍼런스

| 변수 | 필수 | 기본값 | 설명 |
| --- | --- | --- | --- |
| `JOPLIN_TOKEN` | ✅ | — | Joplin Web Clipper API 토큰 |
| `OPENAI_API_KEY` | ✅ | — | OpenAI API 키 |
| `LLM_MODEL` | | `gpt-4o-mini` | 답변 생성 모델 |
| `EMBEDDING_MODEL` | | `text-embedding-3-small` | 임베딩 모델 (변경 시 재인덱싱) |
| `TARGET_NOTE_IDS` | | (전체) | 인덱싱 대상 노트 ID, 콤마 구분 |
| `TARGET_NOTE_ID` | | — | (legacy, 단일 ID. `TARGET_NOTE_IDS` 없을 때만 폴백) |
| `RETRIEVAL_K` | | `10` | 한 쿼리당 회수할 청크 수 |
| `ADMIN_USERS` | | `admin` | 관리자 권한 username, 콤마 구분 |

---

## 12. 운영 체크리스트

- [ ] 새 노트 추가 → `.env` `TARGET_NOTE_IDS` 갱신 → `python -m app.indexer`
- [ ] 기존 노트 수정 → `python -m app.indexer` (증분 자동 감지)
- [ ] 신규 사용자 등록 → 해시 생성 → `auth_config.yaml` 추가 → 서버 재시작
- [ ] 청킹·임베딩 정책 변경 → 인덱스 전체 초기화 후 재인덱싱
- [ ] 매월/매주 `data/usage.db` 또는 admin 사이드바로 비용 확인
- [ ] 정기적으로 `.env`의 `OPENAI_API_KEY`·`JOPLIN_TOKEN` 회전(보안)
