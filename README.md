# ScholarGraph

> Agentic GraphRAG engine combining topological knowledge graph traversal with dense vector search for multi-hop academic reasoning.

ScholarGraph bridges the gap between structured relational data (author networks, paper citations, domain concepts) and unstructured text (paper abstracts and methodology passages). By using **Neo4j** for graph structures, **ChromaDB** for vector retrieval, **Gemini API** for reasoning, and **LangGraph** for dynamic orchestration, ScholarGraph answers complex academic research questions that standard RAG systems fail to address.

---

## Architecture & System Overview

```
+----------------------------------------------------------------------------------+
|                               PHASE 1: INGESTION                                  |
|                                                                                   |
|  [ arXiv API ] ---> Fetch Metadata & Abstracts                                    |
|                           |                                                       |
|                           +---> [ Gemini API ] --------> Extract Graph Triples   |
|                           |                                (Nodes & Edges)        |
|                           |                                       |               |
|                           v                                       v               |
|            [ SentenceTransformers (T4 GPU) ]              [ Neo4j Local ]         |
|                           |                             (Graph DB Ingestion)      |
|                           v                                                       |
|                  [ ChromaDB Local ]                                               |
|                (Vector DB Ingestion)                                              |
+-----------------------------------------------------------------------------------+

+-----------------------------------------------------------------------------------+
|                                PHASE 2: AGENTIC QUERY                             |
|                                                                                   |
|                                [ User Input Query ]                               |
|                                         |                                         |
|                                         v                                         |
|                             [ LangGraph Router Agent ]                            |
|                               (Gemini Decision Loop)                              |
|                                         |                                         |
|                  +----------------------+----------------------+                  |
|                  |                                             |                  |
|                  v                                             v                  |
|       [ Cypher Query Tool ]                         [ Vector Retrieval Tool ]     |
|                 |                                              |                  |
|                 v                                              v                  |
|      (Traverses Neo4j Graph)                       (Searches ChromaDB Chunks)     |
|                 |                                              |                  |
|                 +----------------------+-----------------------+                  |
|                                         |                                         |
|                                         v                                         |
|                             [ Gemini Synthesis Agent ]                            |
|                        (Grounded Answer + Citations)                              |
+-----------------------------------------------------------------------------------+

```

---

## Repository Structure

```text
ScholarGraph/
├── .gitattributes
├── .gitignore
├── pyproject.toml
├── README.md
├── requirements.txt
└── src/
    └── scholargraph/
        ├── __init__.py
        ├── config.py             # Configuration & environment variables
        ├── arxiv_fetcher.py      # arXiv Python API fetching module
        ├── graph_extractor.py    # LLM extraction logic (Gemini + Pydantic)
        ├── embedding_generator.py# Local embedding generation (SentenceTransformers)
        ├── vector_store.py       # ChromaDB interface & operations
        ├── graph_store.py        # Neo4j interface & Cypher execution
        ├── agent.py              # LangGraph orchestration engine
        ├── api.py                # FastAPI HTTP interface for the frontend
        ├── main.py               # Application entry point / CLI interface
        └── tools/
            ├── __init__.py
            ├── cypher_tool.py    # Neo4j graph retrieval tool
            └── vector_tool.py    # ChromaDB vector retrieval tool
    ├── frontend/                     # Independent React + Vite client
    │   └── src/
    │       ├── App.tsx
    │       ├── lib/api.ts            # Typed API client
    │       └── styles.css

```

---

## Tech Stack

| Layer | Technology | Execution Environment |
| --- | --- | --- |
| **LLM Reasoning** | Google Gemini API (`gemini-2.5-flash` / `gemini-2.5-pro`) | Cloud API |
| **Data Ingestion** | `arxiv` Python API & `pydantic` | Google Colab |
| **Embeddings** | `BAAI/bge-small-en-v1.5` | Google Colab (T4 GPU) / Local |
| **Vector Database** | ChromaDB (Persistent Disk Mode) | Local Machine |
| **Graph Database** | Neo4j Community (Docker Container) | Local Machine (`localhost:7687`) |
| **Agent Orchestration** | LangGraph & `google-genai` | Local Machine |

---

1. **Start local Neo4j database**

---

## Data Pipeline & System Execution

### Step 1: Extract & Embed Data (Google Colab / Compute Host)

Run ingestion scripts on Google Colab (with a free T4 GPU enabled) to pull paper abstracts, extract entities via Gemini, and generate text vector embeddings:

1. Fetch abstracts using the arXiv Python API:

```bash
python -m scholargraph.arxiv_fetcher --categories cs.AI cs.CL --max-results 2000

```

2. Extract graph triples (`Paper`, `Author`, `Category`, `Concept`) using Gemini:

```bash
python -m scholargraph.graph_extractor --input data/raw_arxiv.json --output data/graph_triples.json

```

3. Generate embeddings on GPU:

```bash
python -m scholargraph.embedding_generator --input data/raw_arxiv.json --output data/arxiv_vectors.parquet

```

4. Download `graph_triples.json` and `arxiv_vectors.parquet` into your local `data/` folder.

### Step 2: Hydrate Local Databases

Populate your local ChromaDB and Neo4j database instances from the generated data artifacts:

```bash
# Load vector embeddings into ChromaDB
python -m scholargraph.vector_store --import data/arxiv_vectors.parquet

# Ingest graph triples into Neo4j
python -m scholargraph.graph_store --import data/graph_triples.json

```

### Step 3: Run the ScholarGraph Agent

Launch the CLI interface to query your hybrid GraphRAG agent:

```bash
python -m scholargraph.main

```

### Step 4: Run the web application

The Python package remains the backend, and the React client lives in `frontend/`. Start the API from the repository root in one terminal:

```powershell
python -m pip install -r requirements.txt
uvicorn scholargraph.api:app --app-dir src --reload
```

Then start the frontend in a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open <http://localhost:5173>. Vite proxies `/api` requests to the backend at `http://127.0.0.1:8000`. The API uses the root `.env` for Gemini and Neo4j configuration; Neo4j must be running and the graph and vector stores must be populated as described above. Chat history is stored in the browser. For a separately hosted API, set `VITE_API_URL` in the frontend environment and `FRONTEND_ORIGIN` for backend CORS.

---

## Example Interaction

**User Query:**

> *"Which researchers are collaborating on Graph Neural Networks, and what key techniques are they using in their recent abstracts?"*

**Agent Execution Strategy:**

1. **Router Agent:** Evaluates query intent and triggers a combined retrieval path.
2. **Cypher Tool Execution:** Queries Neo4j for co-authorship relationships and paper IDs within the "Graph Neural Networks" category.

```cypher
MATCH (a1:Author)-[:AUTHORED]->(p:Paper)<-[:AUTHORED]-(a2:Author)
MATCH (p)-[:USES_METHOD]->(c:Concept {name: "Graph Neural Networks"})
RETURN a1.name, a2.name, p.id, p.title

```

1. **Vector Tool Execution:** Uses the paper IDs returned by Neo4j to pull exact contextual abstracts and methodologies from ChromaDB.
2. **Synthesis Agent:** Combines structural network paths with semantic text snippets to format a grounded answer complete with arXiv paper citations.

---
