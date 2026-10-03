# ScholarGraph

> A research assistant that combines knowledge-graph traversal and semantic paper search to answer questions about academic literature.

ScholarGraph connects structured paper relationships (authors, categories, and methods) with unstructured abstract text. The Python backend uses **LangGraph** for multi-stage workflow orchestration, **Neo4j** for graph queries, **ChromaDB** for semantic retrieval, **BAAI/bge-reranker-base** for neural reranking, and **Google Gemini** for query translation and grounded answer synthesis. A separate **React** frontend communicates with the backend over HTTP.

---

## Architecture & System Overview

```text
+----------------------------------------------------------------------------------+
|                               PHASE 1: INGESTION                                 |
|                                                                                  |
|  [ OpenAlex API ] ---> Fetch Metadata & Abstracts                                |
|                           |                                                      |
|                           +---> [ Gemini API ] --------> Extract Graph Triples   |
|                           |                               (Nodes & Edges)        |
|                           |                                      |               |
|                           v                                      v               |
|            [ SentenceTransformers (T4 GPU) ]              [ Neo4j Local ]        |
|                           |                             (Graph DB Ingestion)     |
|                           v                                                      |
|                  [ ChromaDB Local ]                                              |
|                (Vector DB Ingestion)                                             |
+-----------------------------------------------------------------------------------+

+-----------------------------------------------------------------------------------+
|                        PHASE 2: LANGGRAPH ORCHESTRATION ENGINE                    |
|                                                                                   |
|                                [ User Input Query ]                               |
|                                          |                                        |
|                                          v                                        |
|                                [ Query Understanding ]                            |
|                            (Gemini Cypher Plan & Search)                          |
|                                          |                                        |
|                        +-----------------+-----------------+                      |
|                        |                                   |                      |
|                        v                                   v                      |
|               [ Graph Retrieval ]                 [ Semantic Retrieval ]          |
|             (Neo4j Cypher & Fallback)             (ChromaDB Dense Search)         |
|                        |                                   |                      |
|                        +-----------------+-----------------+                      |
|                                          |                                        |
|                                          v                                        |
|                                 [ Candidate Fusion ]                              |
|                            (Reciprocal Rank Fusion - RRF)                         |
|                                          |                                        |
|                                          v                                        |
|                                 [ Graph Expansion ]                               |
|                           (1-Hop Subgraph Neighborhoods)                          |
|                                          |                                        |
|                                          v                                        |
|                                [ Cross-Encoder Reranker ]                         |
|                           (BAAI/bge-reranker-base Scoring)                        |
|                                          |                                        |
|                                          v                                        |
|                                 [ Answer Synthesis ]                              |
|                             (Grounded Gemini Generation)                          |
+-----------------------------------------------------------------------------------+
```

---

## Tech Stack

| Layer | Technology | Execution Environment |
| --- | --- | --- |
| **Workflow Orchestration** | LangGraph (`StateGraph`) | Local Python process |
| **LLM Reasoning & Synthesis** | Google Gemini API (`google-genai`) | Cloud API |
| **Data Collection** | OpenAlex-based collection script and JSON paper artifacts | Colab-oriented pipeline / local files |
| **Embeddings** | `BAAI/bge-small-en-v1.5` | Google Colab (T4 GPU) / Local |
| **Neural Reranking** | `BAAI/bge-reranker-base` (Cross-Encoder) | Local Python process / GPU |
| **Vector Database** | ChromaDB (Persistent Disk Mode) | Local Machine |
| **Graph Database** | Neo4j Community (Docker Container) | Local Machine (`localhost:7687`) |
| **Backend API** | FastAPI + Uvicorn | Local Python process |
| **Frontend** | React, TypeScript, Vite | Browser / Node.js development server |

---

## Core Concepts

### GraphRAG & LangGraph Orchestration

Retrieval-Augmented Generation (RAG) grounds language models in structured facts and textual evidence. ScholarGraph orchestrates a multi-step GraphRAG pipeline built with **LangGraph (`StateGraph`)**, which structures state transitions between query processing, dual-channel retrieval, candidate fusion, graph expansion, cross-encoder reranking, and citation synthesis.

### Knowledge Graph

A knowledge graph represents entities as **nodes** and their connections as **relationships**. ScholarGraph models `Paper`, `Author`, `Category`, and `Concept` nodes. Relationships include `AUTHORED`, `IN_CATEGORY`, and `USES_METHOD`. Neo4j stores this structure, and Cypher expresses multi-hop traversals such as "which authors published in this category using a specific AI concept?"

### Candidate Fusion & Graph Expansion

* **Reciprocal Rank Fusion (RRF):** Merges independent candidate rank lists from Neo4j (structural traversal) and ChromaDB (semantic vector search) into a unified score list without requiring normalized similarity scales.
* **1-Hop Neighborhood Graph Expansion:** Takes the fused top-ranked candidate paper IDs and expands their subgraphs in Neo4j to retrieve complete metadata (all co-authors, connected categories, and extracted concepts).

### Cross-Encoder Neural Reranking

Candidate payloads containing paper titles, concepts, and abstract text are scored against the user query using a dedicated Cross-Encoder model (`BAAI/bge-reranker-base`). Unlike bi-encoder embeddings, the cross-encoder performs joint self-attention across the query and document payload simultaneously to accurately re-order the top $N$ candidates by neural relevance.

### Grounded Answer Synthesis

The top reranked evidence payloads are formatted into a strict context prompt for Gemini. Every output statement must cite one or more exact paper IDs present in the allowed evidence. Responses with unsupported citations or hallucinated IDs are automatically flagged and rejected.

---

## Data Pipeline

The pipeline starts with paper metadata and abstracts, enriches each paper with author/category/method relationships, and produces dense embeddings of abstract text. Graph triples are imported into Neo4j; precomputed embeddings are imported into the persistent ChromaDB collection named `arxiv_papers`. The existing `data/README.md` documents the artifact fields and formats. The paper-collection script is Colab-oriented, while local extraction and database import utilities live in `backend/src/scholargraph/`.

The repository may not contain every generated artifact: embedding Parquet files and the persistent Chroma index are local/generated data and should be provisioned for the environment. Neo4j must also be running and hydrated before research queries can return useful results.

---

## Run the Application

The backend and frontend have separate setup and run guides:

- [Backend setup and API guide](backend/README.md)
- [Frontend setup and development guide](frontend/README.md)

For a local chat session, start the FastAPI server from `backend/`, then start Vite from `frontend/`. The frontend is served at <http://localhost:5173> and the API defaults to <http://127.0.0.1:8000>.

---

## Application Screenshots

### Start a conversation

The landing page offers example queries, a query box, and access to previous conversations.

![ScholarGraph landing page](pictures/picture_1.png)

### Example 1: 

![ScholarGraph response about graph neural networks](pictures/picture_2.png)

### Example 2: 

![ScholarGraph response about retrieval-augmented generation](pictures/picture_3.png)

---

## Example Interaction

**User Query:**

> *"Which researchers are collaborating on Graph Neural Networks, and what key techniques are they using in their recent abstracts?"*

**LangGraph Execution Pipeline:**

1. **Query Understanding:** Gemini analyzes user intent and outputs a schema-conforming read-only Cypher query.
2. **Parallel Retrieval:**
   - **Graph Retrieval:** Executes the Cypher query in Neo4j to retrieve co-author nodes and category relationships (with keyword fallback support).
   - **Semantic Retrieval:** Queries ChromaDB for the top semantically relevant paper abstracts.
3. **Candidate Fusion:** Merges paper candidate lists from both stores using Reciprocal Rank Fusion (RRF).
4. **Graph Expansion:** Performs 1-hop Neo4j neighborhood queries for fused paper IDs to pull full author lists, concepts, and categories.
5. **Cross-Encoder Reranking:** Computes neural relevance scores for fused paper payloads against the original prompt using `BAAI/bge-reranker-base` and filters the top results.
6. **Answer Synthesis:** Gemini receives only the top reranked evidence payloads and synthesizes a structured, citation-grounded answer.

```text
MATCH (p:Paper)-[:IN_CATEGORY]->(:Category {name: 'Graph Neural Networks'})
MATCH (a:Author)-[:AUTHORED]->(p)
RETURN DISTINCT a.name AS author, p.id AS paper_id, p.title AS title
ORDER BY author
LIMIT 25
```

---
