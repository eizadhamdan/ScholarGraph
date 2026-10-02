# ScholarGraph Backend

The backend owns data ingestion utilities, graph/vector retrieval, the Gemini-powered research agent, and the HTTP API used by the React frontend. It is a Python 3.12 package under `src/scholargraph/`.

For the overall architecture and the GraphRAG concepts used here, see the [repository guide](../README.md).

## Components

- `src/scholargraph/api.py`: FastAPI application. It exposes conversation CRUD, chat, and health endpoints.
- `src/scholargraph/chat_store.py`: SQLite schema and persistence operations for conversations and ordered messages.
- `src/scholargraph/agent.py`: Runs graph retrieval before vector retrieval, requires structured claims cited with retrieved paper IDs, and returns both an answer and retrieval trace.
- `src/scholargraph/graph_store.py`: Creates the Neo4j driver, imports graph triples, and executes Cypher.
- `src/scholargraph/vector_store.py`: Imports precomputed embeddings to ChromaDB and performs vector retrieval. The embedding model is cached for reuse within the process.
- `src/scholargraph/graph_extractor.py`: Builds author, category, and method relationships from paper JSON; Gemini can enrich the concepts.
- `src/scholargraph/embedding_generator.py`: Generates normalized abstract embeddings using SentenceTransformers.
- `src/scholargraph/arxiv_fetcher.py`: Colab-oriented collection/preprocessing script. It currently imports Google Colab utilities and is not a local command-line fetcher.
- `src/scholargraph/main.py`: Optional interactive CLI using the same agent as the HTTP API.
- `tests/`: API, retrieval-order, citation-validation, and outage tests that do not need live Gemini or database services.

## Requirements

- Python 3.12
- A Gemini API key for chat and optional Gemini-based concept extraction
- Neo4j running and populated with the expected graph schema
- A populated ChromaDB collection named `arxiv_papers`
- SQLite, included with Python, for persistent conversation history

The project root `.env` is loaded by the backend configuration. Create or update it without committing secrets:

```dotenv
GEMINI_API_KEY=your-gemini-key
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-neo4j-password
CHROMA_PERSIST_DIR=./chroma_db
EMBEDDING_MODEL_NAME=BAAI/bge-small-en-v1.5
FRONTEND_ORIGIN=http://localhost:5173
CHAT_HISTORY_DB=./data/chat_history.sqlite3
```

`CHROMA_PERSIST_DIR` is relative to the backend process working directory. The default `./chroma_db` points to this folder's local persistent store. That store is generated/local state and is ignored by Git.

`CHAT_HISTORY_DB` defaults to `backend/data/chat_history.sqlite3`. Relative values are resolved from the backend package root. The database is created and migrated to the current schema when the API starts; its SQLite and WAL files are ignored by Git. This is a single-install store without user authentication, so all clients connected to the same backend share its conversations.

## Install and Run

From the repository root, open a terminal in `backend/`:

```bash
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -e .
```

Start the API from `backend/`:

```bash
python -m uvicorn scholargraph.api:app --reload
```

The API listens at `http://127.0.0.1:8000` by default. Useful routes:

- `GET /api/health` reports API process state and probes Neo4j. It returns `{"status":"ok","knowledge_graph_available":true}` when Neo4j is reachable, or `{"status":"degraded","knowledge_graph_available":false}` when it is not. It does not check Gemini or ChromaDB connectivity.
- `GET /api/conversations` lists saved conversations, newest first.
- `POST /api/conversations` creates an empty conversation.
- `GET /api/conversations/{id}` loads a conversation and its ordered messages.
- `DELETE /api/conversations/{id}` deletes the conversation and its messages.
- `POST /api/chat` accepts JSON such as `{"message":"Find papers about graph neural networks","conversation_id":"..."}` and returns the answer, retrieval trace, and stored user/assistant messages.

Chat input is limited to 4,000 characters. The backend loads up to the latest 12 stored turns as synthesis context; the browser cannot supply or alter that history through the chat request.

Before query planning, the agent probes Neo4j; if it is unavailable, the request is rejected before calling Gemini. If Neo4j becomes unreachable during traversal, `POST /api/chat` returns HTTP `503` and stops. If Gemini's generated Cypher is rejected or returns zero rows, the agent tries a read-only keyword fallback. If that fallback also fails, the API returns HTTP `422` without an answer. The user message is kept in conversation history, but no assistant answer is written for failed turns.

For a successful request, retrieval order is Cypher generation, Neo4j traversal, ChromaDB search (five papers), then answer synthesis. If Gemini's graph query returns zero rows or is rejected, the agent retries Neo4j with parameterized keyword matching over paper titles, methods, and categories. Every generated finding must cite one or more source IDs returned by one of the databases; invented IDs, malformed structured output, and empty evidence are rejected. The saved assistant message includes each Cypher attempt, its result count, error/fallback terms, up to 20 graph rows, plus vector-source titles, IDs, dates, and excerpts. The frontend exposes this trace in the “Retrieval evidence” panel, and the CLI prints graph/vector match counts. Gemini service outages return a retryable HTTP `503` with counts for retrieval completed so far.

The CLI can be run from the same `backend/` working directory:

```bash
python -m scholargraph.main
```

## Data Preparation

Paper artifacts live in the repository-level `data/` directory. From `backend/`, graph triples can be imported with:

```bash
python -m scholargraph.graph_store --import ../data/graph_triples.json
```

To import precomputed vectors, provide the generated Parquet artifact:

```bash
python -m scholargraph.vector_store --import ../data/arxiv_vectors.parquet
```

`arxiv_vectors.parquet` may need to be generated or downloaded separately; it is not guaranteed to be present in a checkout. `graph_extractor.py` can create triples from the raw paper JSON:

```bash
python -m scholargraph.graph_extractor --input ../data/raw_arxiv_papers.json --output ../data/graph_triples.json
```

Embedding generation is intended for a compute host with SentenceTransformers and may use a GPU when available. The current collection script is Colab-oriented. Artifact schemas are described in [`data/README.md`](../data/README.md).

## Tests

From `backend/`, with the project environment active:

```bash
python -m pytest tests -q
```

These tests validate API persistence, graph-before-vector-before-synthesis order, citation validation, and unavailable-graph behavior without calling Gemini or connecting to Neo4j/ChromaDB.

## Deployment Note

Generated Cypher is executed in a Neo4j read transaction, and its prompt is restricted to read queries. Still use a least-privilege Neo4j account before exposing this service to untrusted users or a public network. Citation validation confirms IDs came from retrieved data, not that each passage semantically entails its associated claim; verify important findings against the displayed excerpts and original papers.
