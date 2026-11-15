# Palm Mind RAG API

A FastAPI backend with two REST APIs:

1. **Document Ingestion**: upload `.pdf` / `.txt`, extract text, chunk it with one of two selectable strategies,
   embed the chunks, store the vectors in **Qdrant**, and store the document metadata in **SQL** (SQLite via SQLAlchemy).
2. **Conversational RAG**: a custom retrieval-augmented chat (no `RetrievalQAChain`, no LangChain) with
   **Redis** chat memory, multi-turn query rewriting, and **LLM-driven interview booking** stored in SQL.

Stack: FastAPI · Pydantic v2 · SQLAlchemy 2 · Qdrant · Redis · Google Gemini (chat + embeddings) · pypdf.

## Architecture

```
app/
├── main.py                 # App factory, lifespan (DB + Qdrant collection init), error handlers
├── core/config.py          # Typed settings from environment / .env
├── db/session.py           # SQLAlchemy engine, session, Base
├── models/                 # ORM tables: Document, Booking
├── schemas/                # Pydantic request/response models
├── api/
│   ├── deps.py             # Dependency injection (clients + services)
│   └── routes/             # documents.py, chat.py, bookings.py
└── services/
    ├── text_extraction.py  # PDF/TXT -> text
    ├── chunking.py         # fixed-size & sentence chunking
    ├── llm.py              # Gemini wrapper: embed / generate / structured JSON output
    ├── vector_store.py     # Qdrant wrapper: upsert / search
    ├── ingestion.py        # Ingestion pipeline
    ├── memory.py           # Redis chat history
    ├── booking.py          # Booking persistence
    ├── prompts.py          # Prompt templates
    └── chat.py             # Custom RAG + booking orchestration
```

Routes are thin. Business logic lives in services that receive their dependencies through constructors,
so the LLM, vector store, or memory backend can be swapped or mocked without touching other code.

### Ingestion flow
`upload -> extract text -> chunk (fixed | sentence) -> embed (RETRIEVAL_DOCUMENT) -> Qdrant upsert -> SQL metadata`

### Chat flow (per turn)
1. Load the session's history from Redis.
2. **One structured LLM call** (JSON schema enforced) returns:
   - `intent`: `question`, `booking` (make a new booking) or `booking_status` (ask about existing bookings,
     e.g. "do I have an interview in the next 2 days?")
   - `standalone_question`: the latest message rewritten so it is self-contained (e.g. "what does it cost?" becomes
     "what does the Pro plan cost?"). This is what makes multi-turn retrieval work.
   - `name`, `email`, `date`, `time`: booking fields gathered across the whole conversation.
3. **Booking**: if fields are missing, ask for them. Otherwise validate (email format, date/time in the future,
   no duplicate slot) and save to the `bookings` table.
4. **Booking status**: look up upcoming bookings for this session (or the email the user gives) in SQL. The days
   remaining are computed in code and passed to the LLM as facts, because LLMs are unreliable at date arithmetic.
5. **Question**: embed the standalone question (RETRIEVAL_QUERY), retrieve top-k chunks from Qdrant, and generate
   an answer grounded only in those chunks plus the chat history.
6. Append the user message and assistant answer to Redis (capped length, TTL).

### Retrieval scope
Every upload is attached to a chat session. The response returns its `session_id` (a new one when none is
sent), so uploading a file and then chatting with that `session_id` answers only from that session's files.
The scope is chosen in this order:
1. `document_ids` in the chat request, if given
2. otherwise, the documents uploaded in this session
3. otherwise, every uploaded document

Each scope is applied as a Qdrant payload filter on an indexed `document_id` field.

### Chunking strategies
| Strategy   | How it works | Trade-off |
|------------|--------------|-----------|
| `fixed`    | Sliding window of `CHUNK_SIZE` characters with `CHUNK_OVERLAP` overlap | Predictable sizes, but may cut sentences |
| `sentence` | Whole sentences packed up to `CHUNK_SIZE` characters (very long sentences are split) | Keeps meaning intact, so retrieval is usually better |

## Setup

Prerequisites: Python 3.12+, Docker, and a free Gemini API key from https://aistudio.google.com/apikey

```bash
docker compose up -d                 # Qdrant on :6333, Redis on :6379
python -m venv .venv
.venv\Scripts\activate               # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
cp .env.example .env                 # then put your GEMINI_API_KEY in .env
uvicorn app.main:app --reload
```

Interactive docs: http://localhost:8000/docs

Run tests: `pytest`

Type check (strict): `mypy app --strict --ignore-missing-imports`

## API

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/documents` | Upload a file (`file`, `chunking_strategy` = `fixed` \| `sentence`, optional `session_id`) |
| GET | `/api/v1/documents` | List ingested documents (optional `?session_id=` filter) |
| DELETE | `/api/v1/documents/{id}` | Delete a document's metadata and all its vectors |
| POST | `/api/v1/chat` | Send a message (`message`, optional `session_id`, optional `document_ids`) |
| GET | `/api/v1/chat/{session_id}/history` | Chat history from Redis |
| DELETE | `/api/v1/chat/{session_id}` | Clear a session's history |
| GET | `/api/v1/bookings` | List interview bookings |
| GET | `/health` | Health check |

### Examples

```bash
curl -F "file=@company_handbook.pdf" -F "chunking_strategy=sentence" http://localhost:8000/api/v1/documents

curl -X POST http://localhost:8000/api/v1/chat -H "Content-Type: application/json" \
     -d '{"message": "What is the leave policy?"}'
# -> {"session_id": "4f1c...", "intent": "question", "answer": "...", "sources": [...]}

curl -X POST http://localhost:8000/api/v1/chat -H "Content-Type: application/json" \
     -d '{"session_id": "4f1c...", "message": "I want to book an interview. I am Sita, sita@example.com"}'
# -> asks for the missing date and time

curl -X POST http://localhost:8000/api/v1/chat -H "Content-Type: application/json" \
     -d '{"session_id": "4f1c...", "message": "next Monday at 2pm"}'
# -> {"intent": "booking", "answer": "Your interview is booked, Sita! ...", "booking": {...}}
```

## Security
- **SQL injection**: there is no raw SQL. All queries use SQLAlchemy's query builder, which sends user input as
  bound parameters (`WHERE email = ?`), never as part of the SQL text.
- **Input validation**: Pydantic validates every request (email format, future dates, `session_id` limited to
  `[A-Za-z0-9_-]`, upload type and size limits).
- **Vector store / Redis**: Qdrant filters are structured objects, and Redis keys are passed as literal arguments,
  so neither parses user input as a query language.
- **Static analysis**: `bandit -r app` reports no issues.
- **Known limits**: there is no authentication; booking lookup by email is unverified (production would confirm
  the email with a one-time code); retrieved document text could contain prompt-injection attempts, which the
  system prompt mitigates but cannot fully prevent.

## Design decisions
- **Custom RAG, no LangChain chains**: retrieval, prompt assembly and generation are explicit in `services/chat.py`.
- **Structured output** (Gemini JSON schema mapped to a Pydantic model) for intent and booking extraction, so the LLM's
  output is validated rather than parsed from free text.
- **Query rewriting** gives follow-up questions accurate retrieval in multi-turn conversations.
- **Retrieved context goes in the system prompt, not in memory**, so Redis only stores the real conversation.
- **Asymmetric embeddings**: `RETRIEVAL_DOCUMENT` for chunks and `RETRIEVAL_QUERY` for questions.
- Sync endpoints: FastAPI runs them in a thread pool, which matches the sync SDK clients used.
