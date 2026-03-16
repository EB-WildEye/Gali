# Gali — Microservices Architecture

## From Monolith to Microservices

````carousel
### Before: Monolith
```
Gali/
├── agent/
│   ├── server.py       ← ALL routes in one FastAPI app
│   ├── llm.py          ← Gemini client
│   ├── vectorstore.py  ← LanceDB wrapper
│   ├── history.py      ← MongoDB history
│   ├── config.py       ← Settings
│   ├── prompt.py       ← System prompt
│   └── logger.py       ← Custom logging
├── ingestion/
│   └── main.py         ← CLI script
├── ui/
│   └── app.py          ← Streamlit UI
└── pyproject.toml
```
**One process, one server, everything coupled.**
<!-- slide -->
### After: Microservices
```
Gali/
├── shared/             ← DB API (Lambda Layer)
│   ├── db.py
│   ├── pii.py
│   └── config.py
├── endpoints/          ← 3 API Lambdas
│   ├── session/
│   ├── chat/
│   └── cleanup/
├── data/               ← Ingestion Lambda
│   └── handler.py
└── frontend/           ← Next.js (React)
    └── src/
```
**5 independent services, each deploys separately.**
````

---

## Architecture Overview

```mermaid
flowchart TB
    subgraph FRONTEND["frontend/ — AWS Amplify"]
        NEXT["Next.js App"]
    end

    subgraph ENDPOINTS["endpoints/ — API Lambdas"]
        L1["λ Session<br/>handler.py"]
        L2["λ Chat<br/>handler.py + llm/vectorstore/prompt"]
        L4["λ Cleanup<br/>handler.py"]
    end

    subgraph DATA["data/ — Ingestion Lambda"]
        L3["λ Ingestion<br/>handler.py"]
    end

    subgraph SHARED["shared/ — Lambda Layer"]
        DB["db.py"]
        PII["pii.py"]
        CONF["config.py"]
    end

    subgraph INFRA["AWS Infrastructure"]
        GW["API Gateway"]
        S3D["S3 gali-documents"]
        S3V["S3 gali-vectors"]
        ATLAS["MongoDB Atlas"]
        SM["Secrets Manager"]
        EB["EventBridge"]
    end

    PATIENT["Patient 🖥️"] --> NEXT
    NEXT --> GW
    GW --> L1
    GW --> L2
    S3D -->|PutObject| L3
    EB -->|rate 1h| L4

    L1 -.-> DB
    L2 -.-> DB
    L4 -.-> DB
    L3 -.-> CONF

    L1 --> ATLAS
    L2 --> ATLAS
    L2 --> S3V
    L2 --> GEMINI["Gemini API"]
    L3 --> S3V
    L3 --> GEMINI
    L4 --> ATLAS
    SM -.-> CONF
```

---

## Full Folder Structure

```
Gali/
│
├── shared/                          ← SHARED DATA LAYER (Lambda Layer)
│   ├── db.py                        ← MongoDB connection + all DB operations
│   ├── pii.py                       ← PII scrubbing (IDs, phones, emails)
│   └── config.py                    ← Secrets Manager helper + env settings
│
├── endpoints/                       ← API LAMBDAS (future repo: gali-api)
│   │
│   ├── session/                     ← λ1 — Session Service
│   │   ├── handler.py               ← GET /session/new + GET /history/{id}
│   │   └── requirements.txt         ← aws-lambda-powertools, pymongo
│   │
│   ├── chat/                        ← λ2 — Chat Agent Service
│   │   ├── handler.py               ← POST /chat (RAG pipeline)
│   │   ├── requirements.txt         ← powertools, google-genai, lancedb, pymongo
│   │   ├── llm.py                   ← Gemini client (embed + generate)
│   │   ├── vectorstore.py           ← LanceDB search
│   │   └── prompt.py                ← System prompt (Gali persona)
│   │
│   └── cleanup/                     ← λ4 — Cleanup Service
│       ├── handler.py               ← Scheduled: delete messages > 24h
│       └── requirements.txt         ← aws-lambda-powertools, pymongo
│
├── data/                            ← INGESTION LAMBDA (future repo: gali-data)
│   ├── handler.py                   ← S3 trigger: PDF → extract → chunk → embed → store
│   └── requirements.txt             ← powertools, pdfplumber, google-genai, lancedb
│
└── frontend/                        ← NEXT.JS APP (future repo: gali-frontend)
    ├── package.json
    ├── next.config.js
    └── src/
        ├── app/
        │   ├── layout.tsx           ← RTL, fonts, global styles
        │   └── page.tsx             ← Main chat page
        ├── components/
        │   ├── ChatBubble.tsx       ← Message bubble component
        │   ├── ChatInput.tsx        ← Input bar + send button
        │   ├── Header.tsx           ← Green header "גלי"
        │   └── Sidebar.tsx          ← About + disclaimer
        └── lib/
            └── api.ts               ← API Gateway call functions
```

---

## UML — Shared Layer

```mermaid
classDiagram
    class db {
        -_client : MongoClient
        -_col : Collection
        +get_collection() Collection
        +save_turn(session_id, user_msg, assistant_msg) void
        +get_messages(session_id, limit) list~dict~
        +get_chat_history(session_id) list~dict~
        +delete_old_messages(hours) int
        +check_connection() void
    }

    class pii {
        -_PII_PATTERNS : list~tuple~
        +remove_pii(text) str
    }

    class config {
        +get_mongo_uri() str
        +get_gemini_key() str
        +get_setting(name, default) str
    }

    db --> pii : uses remove_pii before saving
    db --> config : uses get_mongo_uri()
```

### What each function does:

| File | Function | Description |
|------|----------|-------------|
| **db.py** | `get_collection()` | Returns MongoDB `chat_history` collection (lazy connects) |
| | `save_turn(sid, user, bot)` | Scrubs PII → adds `created_at` timestamp → inserts both messages |
| | `get_messages(sid, limit=20)` | Returns last N messages oldest-first `[{role, content}]` |
| | `get_chat_history(sid)` | Returns messages in Gemini format `[{role: "model", parts: [...]}]` |
| | `delete_old_messages(hours=24)` | Deletes messages where `created_at < now - hours` → returns count |
| | `check_connection()` | Pings MongoDB to verify connectivity |
| **pii.py** | `remove_pii(text)` | Regex-scrubs Israeli IDs (9 digits), phone numbers, emails |
| **config.py** | `get_mongo_uri()` | Fetches `/gali/MONGO_URI` from Secrets Manager (cached 5 min) |
| | `get_gemini_key()` | Fetches `/gali/GEMINI_API_KEY` from Secrets Manager (cached 5 min) |
| | `get_setting(name, default)` | Reads env var with fallback |

---

## UML — Session Service (λ1)

```mermaid
classDiagram
    class handler {
        -logger : Logger
        -app : APIGatewayHttpResolver
        +create_session() dict
        +retrieve_session_history(session_id) list
        +lambda_handler(event, context) dict
    }

    handler --> db : get_messages()
```

| Function | Route | Description |
|----------|-------|-------------|
| `create_session()` | `GET /session/new` | Generates UUID v4, returns `{session_id}` |
| `retrieve_session_history()` | `GET /history/<session_id>` | Calls `shared.db.get_messages()` → returns messages list |
| `lambda_handler()` | — | Entry point, resolves API Gateway event to route |

---

## UML — Chat Agent Service (λ2)

```mermaid
classDiagram
    class handler {
        -logger : Logger
        -tracer : Tracer
        -app : APIGatewayHttpResolver
        -_llm : GeminiClient
        -_store : VectorStore
        +_init_services() void
        +sanitize_user_input(text) str
        +handle_chat_message(body: ChatRequest) ChatResponse
        +handle_unexpected_error(ex) Response
        +lambda_handler(event, context) dict
    }

    class ChatRequest {
        +query : str
        +session_id : str
    }

    class ChatResponse {
        +answer : str
        +sources : list~str~
        +session_id : str
    }

    class GeminiClient {
        -_client : genai.Client
        -_config : GenerateContentConfig
        +embed_text(text) list~float~
        +generate_answer(query, context, history) str
    }

    class VectorStore {
        -_db : lancedb.Connection
        -_llm : GeminiClient
        +search_similar(query, top_k) list~dict~
    }

    handler --> GeminiClient : _init_services creates
    handler --> VectorStore : _init_services creates
    handler --> db : save_turn(), get_chat_history()
    handler ..> ChatRequest : validates input
    handler ..> ChatResponse : returns
    VectorStore --> GeminiClient : uses embed_text()
    GeminiClient --> prompt : reads SYSTEM_PROMPT
```

| File | Function | Description |
|------|----------|-------------|
| **handler.py** | `_init_services()` | Lazy-init GeminiClient + VectorStore (reused across warm invocations) |
| | `sanitize_user_input(text)` | NFKC normalize → strip invisible chars → detect injection patterns |
| | `handle_chat_message(body)` | Full RAG: sanitize → search → get history → generate → save → respond |
| **llm.py** | `GeminiClient.__init__()` | Connects to Gemini API, sets system_instruction |
| | `embed_text(text)` | Returns 768-dim embedding vector |
| | `generate_answer(query, ctx, history)` | Creates chat with history → sends context+query → retries on 503/429 |
| **vectorstore.py** | `VectorStore.__init__(llm)` | Connects to LanceDB (S3 path) |
| | `search_similar(query, top_k=4)` | Embeds query → searches LanceDB → returns top-k chunks |
| **prompt.py** | `SYSTEM_PROMPT` | Gali persona: Hebrew medical coordinator, triage rules, disclaimers |

---

## UML — Ingestion Service (λ3)

```mermaid
classDiagram
    class handler {
        -logger : Logger
        -tracer : Tracer
        -s3 : boto3.Client
        -_embed_client : EmbedClient
        -_store : lancedb.Connection
        +init_services() void
        +normalize_text(text) str
        +extract_pdf_text(file_bytes) str
        +extract_csv_text(file_bytes) str
        +split_into_chunks(text, source) list~dict~
        +lambda_handler(event, context) dict
    }

    class EmbedClient {
        -_client : genai.Client
        -_model : str
        +embed_text(text) list~float~
    }

    handler --> EmbedClient : embeds chunks
    handler --> config : get_gemini_key()
```

| Function | Description |
|----------|-------------|
| `lambda_handler(event)` | Reads S3 event → downloads file → processes → stores in LanceDB |
| `init_services()` | Lazy-init Gemini embed client + LanceDB connection |
| `extract_pdf_text(bytes)` | pdfplumber: extract text + tables from PDF bytes |
| `extract_csv_text(bytes)` | Decode CSV bytes to cleaned text |
| `normalize_text(text)` | Strip junk chars, collapse whitespace, remove bullets |
| `split_into_chunks(text, source)` | Split into 800-char chunks with 200-char overlap |

---

## UML — Cleanup Service (λ4)

```mermaid
classDiagram
    class handler {
        -logger : Logger
        +lambda_handler(event, context) dict
    }

    handler --> db : delete_old_messages(24)
```

| Function | Description |
|----------|-------------|
| `lambda_handler(event)` | Calls `shared.db.delete_old_messages(24)` → logs count → returns |

---

## UML — Frontend

```mermaid
classDiagram
    class api_ts {
        +createSession() Promise~string~
        +getHistory(sessionId) Promise~Message[]~
        +sendMessage(query, sessionId) Promise~ChatResponse~
    }

    class page_tsx {
        -sessionId : string
        -messages : Message[]
        +handleSend(text) void
        +loadHistory() void
    }

    class ChatBubble {
        +role : string
        +content : string
        +sources : string[]
    }

    class ChatInput {
        +onSend(text) callback
        +placeholder : string
    }

    class Header {
        +title : string
        +subtitle : string
    }

    page_tsx --> api_ts : calls API
    page_tsx --> ChatBubble : renders messages
    page_tsx --> ChatInput : captures input
    page_tsx --> Header : renders header
```

| File | Function | Description |
|------|----------|-------------|
| **api.ts** | `createSession()` | `GET /session/new` → returns session UUID |
| | `getHistory(sid)` | `GET /history/{sid}` → returns messages array |
| | `sendMessage(query, sid)` | `POST /chat` → returns `{answer, sources}` |
| **page.tsx** | `handleSend(text)` | Appends user message → calls API → appends bot response |
| | `loadHistory()` | On mount: creates session or loads existing history |
| **ChatBubble** | — | Renders single message (RTL, Hebrew, markdown, source tags) |
| **ChatInput** | — | Input bar with send button, placeholder in Hebrew |
| **Header** | — | Green banner: "גלי — עוזרת AI גינקולוגיה" |

---

## Flow Diagrams

### Chat Flow (λ2)

```mermaid
sequenceDiagram
    participant UI as Frontend
    participant GW as API Gateway
    participant Chat as λ2 Chat
    participant Shared as shared/db
    participant Lance as LanceDB (S3)
    participant Gemini as Gemini API
    participant Mongo as MongoDB Atlas

    UI->>GW: POST /chat {query, session_id}
    GW->>Chat: invoke
    Chat->>Chat: sanitize_user_input(query)
    Chat->>Lance: search_similar(query)
    Lance-->>Chat: top 4 chunks
    Chat->>Shared: get_chat_history(session_id)
    Shared->>Mongo: find({session_id})
    Mongo-->>Shared: messages[]
    Shared-->>Chat: Gemini-format history
    Chat->>Gemini: chat.send_message(context + query)
    Gemini-->>Chat: answer
    Chat->>Shared: save_turn(session_id, query, answer)
    Shared->>Shared: remove_pii(text)
    Shared->>Mongo: insert_many([user, assistant])
    Chat-->>GW: {answer, sources}
    GW-->>UI: JSON response
```

### Ingestion Flow (λ3)

```mermaid
sequenceDiagram
    participant Admin as Admin 👤
    participant S3D as S3 gali-documents
    participant Ingest as λ3 Ingestion
    participant Gemini as Gemini API
    participant S3V as S3 gali-vectors

    Admin->>S3D: Upload Protocol.pdf
    S3D->>Ingest: PutObject event
    Ingest->>S3D: Download file bytes
    Ingest->>Ingest: extract_pdf_text()
    Ingest->>Ingest: normalize_text()
    Ingest->>Ingest: split_into_chunks()
    loop Each chunk
        Ingest->>Gemini: embed_text(chunk)
        Gemini-->>Ingest: vector[768]
    end
    Ingest->>S3V: table.add(chunks)
    Ingest-->>Admin: ✅ 54 chunks stored
```

### Cleanup Flow (λ4)

```mermaid
sequenceDiagram
    participant EB as EventBridge
    participant Clean as λ4 Cleanup
    participant Shared as shared/db
    participant Mongo as MongoDB Atlas

    EB->>Clean: Scheduled trigger (hourly)
    Clean->>Shared: delete_old_messages(24)
    Shared->>Mongo: delete_many({created_at < 24h ago})
    Mongo-->>Shared: deleted_count
    Shared-->>Clean: count
    Clean->>Clean: logger.info(deleted: count)
```

---

## File Migration Map

| Old (Monolith) | New (Microservices) | Change |
|----------------|---------------------|--------|
| `agent/server.py` | `endpoints/*/handler.py` | **Split** into 3 Lambda handlers |
| `agent/logger.py` | — | **Deleted** — replaced by Powertools `Logger` |
| `agent/config.py` | `shared/config.py` | **Moved** — reads from Secrets Manager |
| `agent/llm.py` | `endpoints/chat/llm.py` | **Moved** — only chat service needs it |
| `agent/vectorstore.py` | `endpoints/chat/vectorstore.py` | **Moved** — only chat service needs it |
| `agent/history.py` | `shared/db.py` | **Refactored** — becomes shared DB API |
| `agent/prompt.py` | `endpoints/chat/prompt.py` | **Moved** — only chat service needs it |
| `ingestion/main.py` | `data/handler.py` | **Rewritten** — S3 trigger instead of CLI |
| `ui/app.py` | `frontend/src/` | **Rewritten** — Next.js replaces Streamlit |
| `pyproject.toml` | `*/requirements.txt` | **Split** — each service has its own deps |

---

## Technology Stack Per Service

| Service | Runtime | Framework | Dependencies |
|---------|---------|-----------|-------------|
| **shared** | Python 3.12 | — | pymongo, aws-lambda-powertools |
| **session** | Python 3.12 | Powertools `APIGatewayHttpResolver` | pymongo |
| **chat** | Python 3.12 | Powertools `APIGatewayHttpResolver` | google-genai, lancedb, pymongo |
| **ingestion** | Python 3.12 | Powertools `Logger` + `Tracer` | pdfplumber, google-genai, lancedb, boto3 |
| **cleanup** | Python 3.12 | Powertools `Logger` | pymongo |
| **frontend** | Node 20 | Next.js 14 (React) | — |

---

## Future Roadmap: MongoDB → DynamoDB

> The initial deployment uses **MongoDB Atlas** for chat history.
> A future iteration will migrate to **AWS DynamoDB** for full AWS-native integration.

### Why DynamoDB?

| | MongoDB Atlas | DynamoDB |
|---|---|---|
| **Hosting** | External service (MongoDB Inc.) | Native AWS — same account as Lambdas |
| **Connection** | URI string + pymongo driver | `boto3.resource("dynamodb")` — no connection needed |
| **Auth** | Username/password in Secrets Manager | **IAM role** — Lambda has access automatically |
| **Scaling** | Manual tier upgrades | Auto-scales (on-demand mode) |
| **TTL cleanup** | Need λ4 Cleanup Lambda (hourly cron) | **Built-in TTL** — DynamoDB auto-deletes expired items |
| **Cost** | Atlas free tier (512MB) | AWS free tier (25GB + 25 read/write units) |
| **Cold start** | pymongo connection overhead | Zero — boto3 is pre-installed in Lambda |

### What changes in code

Only **`shared/db.py`** changes. No other service is affected (that's the whole point of the shared layer).

```python
# Before (MongoDB)
from pymongo import MongoClient
client = MongoClient(mongo_uri)
col = client["gali"]["chat_history"]
col.insert_many(docs)
col.find({"session_id": sid})

# After (DynamoDB)
import boto3
table = boto3.resource("dynamodb").Table("gali-chat-history")
table.put_item(Item={...})
table.query(KeyConditionExpression=Key("session_id").eq(sid))
```

### DynamoDB table design

```
Table: gali-chat-history
├── Partition Key: session_id  (String)
├── Sort Key:      created_at  (Number — Unix timestamp)
├── Attributes:
│   ├── role       (String — "user" | "assistant")
│   ├── content    (String — message text)
│   └── expires_at (Number — Unix timestamp, TTL attribute)
└── TTL: enabled on "expires_at"
```

### λ4 Cleanup gets deleted

With DynamoDB TTL, expired messages are auto-deleted — no Lambda needed:

```python
# When saving a message, set expires_at = now + 24 hours
import time
item = {
    "session_id": sid,
    "created_at": int(time.time()),
    "role": "user",
    "content": text,
    "expires_at": int(time.time()) + 86400,  # 24h from now
}
table.put_item(Item=item)
# DynamoDB automatically deletes this item after 24 hours
```

### Architecture after DynamoDB migration

```mermaid
flowchart TB
    subgraph FRONTEND["frontend/ — Amplify"]
        NEXT["Next.js"]
    end

    subgraph ENDPOINTS["endpoints/ — 2 Lambdas"]
        L1["λ Session"]
        L2["λ Chat Agent"]
    end

    subgraph DATA["data/ — 1 Lambda"]
        L3["λ Ingestion"]
    end

    subgraph SHARED["shared/"]
        DB["db.py (DynamoDB)"]
    end

    subgraph AWS["AWS Native"]
        GW["API Gateway"]
        DYNAMO["DynamoDB<br/>+ TTL auto-delete"]
        S3D["S3 documents"]
        S3V["S3 vectors"]
    end

    PATIENT["Patient 🖥️"] --> NEXT --> GW
    GW --> L1 & L2
    S3D -->|PutObject| L3

    L1 & L2 -.-> DB
    DB --> DYNAMO
    L2 --> S3V
    L2 & L3 --> GEMINI["Gemini API"]
    L3 --> S3V
```

**Result**: 3 Lambdas instead of 4, zero external dependencies (no Atlas), fully AWS-native.
