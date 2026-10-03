# LangGraph Orchestration Engine with Reciprocal Rank Fusion & Reranking
import json
import re
from dataclasses import dataclass
from typing import Any, TypedDict

from google import genai
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from sentence_transformers import CrossEncoder

from .config import GEMINI_API_KEY, load_gemini_model_catalog
from .gemini import generate_content_with_retry
from .graph_store import (
    GraphQueryError,
    GraphUnavailableError,
    check_graph_connection,
    run_cypher_query,
)
from .vector_store import get_documents_by_ids, query_vector_store


# =====================================================================
# Exceptions & Data Structures
# =====================================================================


class InsufficientEvidenceError(RuntimeError):
    pass


class ModelUnavailableError(RuntimeError):
    def __init__(self, message: str, retrieval: dict[str, Any] | None = None):
        super().__init__(message)
        self.retrieval = retrieval


class GroundingValidationError(RuntimeError):
    pass


class GroundedClaim(BaseModel):
    statement: str = Field(
        min_length=1,
        description=(
            "A substantive finding in plain prose, 2-4 sentences: what the work does, "
            "how it does it, and what it shows. Do not put paper IDs or bracketed "
            "citations in this text."
        ),
    )
    source_ids: list[str] = Field(
        min_length=1,
        description="IDs of the papers supporting this statement, from the allowed list only.",
    )


class GroundedResponse(BaseModel):
    claims: list[GroundedClaim] = Field(min_length=1)


@dataclass
class AgentResult:
    answer: str
    retrieval: dict[str, Any]


# =====================================================================
# LangGraph State Definition
# =====================================================================


class ScholarGraphState(TypedDict, total=False):
    user_query: str
    model_name: str
    conversation_history: list[dict[str, str]]
    cypher_query: str
    semantic_query: str
    graph_raw_results: list[dict[str, Any]]
    graph_query_attempts: list[dict[str, Any]]
    vector_raw_results: list[dict[str, Any]]
    fused_paper_ids: list[str]
    expanded_graph_context: dict[str, dict[str, Any]]
    reranked_evidence: list[dict[str, Any]]
    source_ids: set[str]
    retrieval: dict[str, Any]
    result: AgentResult


# =====================================================================
# Constants & Utility Functions
# =====================================================================

FALLBACK_GRAPH_QUERY = """
MATCH (p:Paper)
OPTIONAL MATCH (p)-[:USES_METHOD]->(method:Concept)
OPTIONAL MATCH (p)-[:IN_CATEGORY]->(category:Category)
WITH p,
     collect(DISTINCT method.name) AS methods,
     collect(DISTINCT category.name) AS categories
WITH p, methods, categories,
     [term IN $terms WHERE
        toLower(coalesce(p.title, '')) CONTAINS term OR
        any(name IN methods WHERE toLower(name) CONTAINS term) OR
        any(name IN categories WHERE toLower(name) CONTAINS term)
     ] AS matched_terms
WHERE size(matched_terms) > 0
RETURN p.id AS paper_id, p.title AS title, methods, categories, matched_terms
ORDER BY size(matched_terms) DESC, p.id
LIMIT 30
"""

EXPANSION_GRAPH_QUERY = """
MATCH (p:Paper)
WHERE p.id IN $paper_ids
OPTIONAL MATCH (a:Author)-[:AUTHORED]->(p)
OPTIONAL MATCH (p)-[:USES_METHOD]->(m:Concept)
OPTIONAL MATCH (p)-[:IN_CATEGORY]->(c:Category)
RETURN p.id AS paper_id,
       p.title AS title,
       collect(DISTINCT a.name) AS authors,
       collect(DISTINCT m.name) AS concepts,
       collect(DISTINCT c.name) AS categories
"""

STOP_WORDS = {
    "about",
    "after",
    "also",
    "and",
    "approach",
    "approaches",
    "are",
    "current",
    "describe",
    "does",
    "find",
    "from",
    "have",
    "into",
    "paper",
    "papers",
    "that",
    "the",
    "their",
    "this",
    "through",
    "what",
    "which",
    "with",
}


# How many claims the synthesis step is asked to write. Raise these (or the 2-4
# sentence guidance in the prompt) for longer answers, lower them for shorter ones.
SYNTHESIS_MIN_CLAIMS = 4
SYNTHESIS_MAX_CLAIMS = 7

OPENALEX_WORK_ID = re.compile(r"^W\d+$")


def _fallback_terms(user_query: str) -> list[str]:
    normalized = re.sub(r"[^a-z0-9]+", " ", user_query.lower())
    terms = [
        token
        for token in normalized.split()
        if len(token) >= 3 and token not in STOP_WORDS
    ]
    return list(dict.fromkeys(terms))[:10]


def reciprocal_rank_fusion(
    graph_paper_ids: list[str],
    vector_paper_ids: list[str],
    rrf_k: int = 60,
    top_n: int = 20,
) -> list[str]:
    """Combines ranked lists from Graph and Vector retrieval via Reciprocal Rank Fusion.

    Each list contributes at most once per paper, at the paper's best rank. Graph
    queries often return one row per author or concept, so the same paper ID can
    repeat; counting every repeat would inflate that paper's score.
    """
    scores: dict[str, float] = {}

    # dict.fromkeys drops repeats but keeps first-occurrence order.
    for rank, paper_id in enumerate(dict.fromkeys(graph_paper_ids)):
        scores[paper_id] = scores.get(paper_id, 0.0) + (1.0 / (rrf_k + rank + 1))

    for rank, paper_id in enumerate(dict.fromkeys(vector_paper_ids)):
        scores[paper_id] = scores.get(paper_id, 0.0) + (1.0 / (rrf_k + rank + 1))

    sorted_candidates = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [paper_id for paper_id, _ in sorted_candidates[:top_n]]


def _short_authors(authors: list[str]) -> str:
    names = [name for name in authors if name]
    if len(names) <= 2:
        return " and ".join(names)
    return f"{names[0]} et al."


def _publication_year(published: Any) -> str:
    year = str(published or "")[:4]
    return year if year.isdigit() else ""


def _format_source_line(number: int, paper_id: str, paper: dict[str, Any]) -> str:
    title = (paper.get("title") or "").strip() or "Untitled paper"
    line = f"{number}. {title}"
    details = ", ".join(
        part
        for part in (
            _short_authors(paper.get("authors") or []),
            _publication_year(paper.get("published")),
        )
        if part
    )
    if details:
        line += f" ({details})"
    if OPENALEX_WORK_ID.match(paper_id):
        line += f" — [OpenAlex](https://openalex.org/{paper_id})"
    return line


def format_grounded_answer(
    claims: list[GroundedClaim], evidence: list[dict[str, Any]]
) -> str:
    """Renders validated claims as readable Markdown with numbered citations.

    Paper IDs stay in the structured claims, where they are validated. The reader
    sees [1], [2], ... in the text and a Sources list with titles instead.
    """
    papers = {item["paper_id"]: item for item in evidence}
    numbers: dict[str, int] = {}
    blocks: list[str] = []

    for index, claim in enumerate(claims):
        labels = []
        for source_id in dict.fromkeys(claim.source_ids):
            number = numbers.setdefault(source_id, len(numbers) + 1)
            labels.append(f"[{number}]")
        text = f"{claim.statement.strip()} {' '.join(labels)}"
        # The first claim is the overview paragraph; the rest are bullet points.
        blocks.append(text if index == 0 else f"- {text}")

    source_lines = [
        _format_source_line(number, source_id, papers.get(source_id, {}))
        for source_id, number in numbers.items()
    ]
    return "\n\n".join([*blocks, "**Sources**", "\n".join(source_lines)])


# =====================================================================
# LangGraph Agent Implementation
# =====================================================================


class ScholarGraphAgent:
    def __init__(self):
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        # BAAI BGE Reranker cross-encoder for precise neural relevance scoring
        self.reranker = CrossEncoder("BAAI/bge-reranker-base")
        self.app = self._build_graph()

    def _build_graph(self):
        """Constructs and compiles the complete LangGraph DAG pipeline."""
        workflow = StateGraph(ScholarGraphState)

        # Nodes
        workflow.add_node("query_understanding", self._node_query_understanding)
        workflow.add_node("graph_retrieval", self._node_graph_retrieval)
        workflow.add_node("semantic_retrieval", self._node_semantic_retrieval)
        workflow.add_node("candidate_fusion", self._node_candidate_fusion)
        workflow.add_node("graph_expansion", self._node_graph_expansion)
        workflow.add_node("reranker", self._node_reranker)
        workflow.add_node("synthesis", self._node_synthesis)

        # Edges
        workflow.add_edge(START, "query_understanding")
        workflow.add_edge("query_understanding", "graph_retrieval")
        workflow.add_edge("graph_retrieval", "semantic_retrieval")
        workflow.add_edge("semantic_retrieval", "candidate_fusion")
        workflow.add_edge("candidate_fusion", "graph_expansion")
        workflow.add_edge("graph_expansion", "reranker")
        workflow.add_edge("reranker", "synthesis")
        workflow.add_edge("synthesis", END)

        return workflow.compile()

    # -----------------------------------------------------------------
    # LangGraph Nodes
    # -----------------------------------------------------------------

    def _node_query_understanding(self, state: ScholarGraphState) -> dict[str, Any]:
        """Node 1: Parses query intent and outputs a structured Cypher query and search terms."""
        if not check_graph_connection():
            raise GraphUnavailableError(
                "Neo4j is unavailable; refusing to generate a response."
            )

        user_query = state["user_query"]
        print(f"\n[1. Query Understanding] Analyzing user prompt: '{user_query}'...")

        schema_prompt = """
        You are a Cypher query generator for a Neo4j database with the following schema:
        - Nodes:
          - (:Paper {id: String, title: String})
          - (:Author {name: String})
          - (:Concept {name: String})
          - (:Category {name: String})
        - Relationships:
          - (:Author)-[:AUTHORED]->(:Paper)
          - (:Paper)-[:USES_METHOD]->(:Concept)
          - (:Paper)-[:IN_CATEGORY]->(:Category)

        Return one read-only Cypher query without markdown formatting or code blocks.
        Never use CREATE, MERGE, DELETE, SET, REMOVE, DROP, or CALL procedures.
        Return paper_id, title, authors, concepts, or categories and limit results to 30 rows.
        """
        try:
            response = generate_content_with_retry(
                self.client,
                model=state.get(
                    "model_name", load_gemini_model_catalog()["default_model"]
                ),
                contents=f"{schema_prompt}\nUser Query: {user_query}",
            )
        except Exception as error:
            raise ModelUnavailableError(
                "Gemini is unavailable while planning the graph query."
            ) from error

        cypher = response.text.replace("```cypher", "").replace("```", "").strip()
        print(f"[Graph Query] Generated Cypher:\n  {cypher}")
        return {"cypher_query": cypher, "semantic_query": user_query}

    def _node_graph_retrieval(self, state: ScholarGraphState) -> dict[str, Any]:
        """Node 2: Executes Cypher graph traversal against Neo4j with keyword fallbacks."""
        cypher = state["cypher_query"]
        user_query = state["user_query"]

        print("[2. Graph Retrieval] Running Neo4j traversal...")
        query_attempts = []
        try:
            graph_results = run_cypher_query(cypher)
            query_attempts.append(
                {
                    "kind": "Gemini-generated",
                    "cypher": cypher,
                    "result_count": len(graph_results),
                }
            )
        except GraphQueryError as error:
            graph_results = []
            query_attempts.append(
                {
                    "kind": "Gemini-generated",
                    "cypher": cypher,
                    "result_count": 0,
                    "error": str(error),
                }
            )

        if not graph_results:
            terms = _fallback_terms(user_query)
            if terms:
                graph_results = run_cypher_query(FALLBACK_GRAPH_QUERY, {"terms": terms})
                query_attempts.append(
                    {
                        "kind": "keyword fallback",
                        "cypher": FALLBACK_GRAPH_QUERY,
                        "result_count": len(graph_results),
                        "parameters": {"terms": terms},
                    }
                )
                print(f"[Graph Tool] Fallback retrieved {len(graph_results)} rows.")

        print(f"[Graph Tool] Retrieved {len(graph_results)} candidate records.")
        return {
            "graph_raw_results": graph_results,
            "graph_query_attempts": query_attempts,
        }

    def _node_semantic_retrieval(self, state: ScholarGraphState) -> dict[str, Any]:
        """Node 3: Executes vector similarity search against ChromaDB."""
        semantic_query = state.get("semantic_query", state["user_query"])
        print("[3. Semantic Retrieval] Performing dense vector search in ChromaDB...")
        vector_results = query_vector_store(semantic_query, n_results=15)
        print(f"[Vector Tool] Retrieved {len(vector_results)} paper candidates.")
        return {"vector_raw_results": vector_results}

    def _node_candidate_fusion(self, state: ScholarGraphState) -> dict[str, Any]:
        """Node 4: Candidate Fusion using Reciprocal Rank Fusion (RRF)."""
        graph_raw = state.get("graph_raw_results", [])
        vector_raw = state.get("vector_raw_results", [])

        print("[4. Candidate Fusion] Running Reciprocal Rank Fusion (RRF)...")

        # Extract paper IDs preserving retrieved order
        graph_ids = []
        for row in graph_raw:
            if isinstance(row, dict):
                p_id = row.get("paper_id") or row.get("p.id") or row.get("id")
                if p_id:
                    graph_ids.append(str(p_id))

        vector_ids = [
            str(item["paper_id"])
            for item in vector_raw
            if item.get("paper_id") is not None
        ]

        fused_ids = reciprocal_rank_fusion(graph_ids, vector_ids, rrf_k=60, top_n=15)
        print(f"[Fusion] Fused top candidates: {fused_ids}")

        if not fused_ids:
            raise InsufficientEvidenceError(
                "Candidate fusion yielded no valid paper matches from graph or vector stores."
            )

        return {"fused_paper_ids": fused_ids}

    def _node_graph_expansion(self, state: ScholarGraphState) -> dict[str, Any]:
        """Node 5: Fetches 1-hop subgraphs (authors, methods, categories) for fused candidate papers."""
        fused_ids = state["fused_paper_ids"]
        print(
            f"[5. Graph Expansion] Expanding 1-hop subgraphs for {len(fused_ids)} papers..."
        )

        expanded_records = run_cypher_query(
            EXPANSION_GRAPH_QUERY, {"paper_ids": fused_ids}
        )

        expanded_map = {}
        for record in expanded_records:
            pid = str(record["paper_id"])
            expanded_map[pid] = {
                "title": record.get("title", ""),
                "authors": record.get("authors", []),
                "concepts": record.get("concepts", []),
                "categories": record.get("categories", []),
            }

        return {"expanded_graph_context": expanded_map}

    def _node_reranker(self, state: ScholarGraphState) -> dict[str, Any]:
        """Node 6: Cross-Encoder neural reranking over enriched paper payloads."""
        user_query = state["user_query"]
        fused_ids = state["fused_paper_ids"]
        vector_raw = state.get("vector_raw_results", [])
        expanded_graph = state.get("expanded_graph_context", {})

        print(
            f"[6. Reranker] Scoring and reranking top candidates using Cross-Encoder..."
        )

        vector_map = {
            str(v["paper_id"]): v.get("document", "")
            for v in vector_raw
            if v.get("paper_id") is not None
        }
        metadata_map = {
            str(v["paper_id"]): v.get("metadata") or {}
            for v in vector_raw
            if v.get("paper_id") is not None
        }

        # Candidates found only through the graph have no vector hit, so they have no
        # abstract yet. Load those abstracts from ChromaDB by ID so the reranker and
        # the synthesis step see the same kind of evidence for every candidate.
        missing_abstract_ids = [pid for pid in fused_ids if not vector_map.get(pid)]
        if missing_abstract_ids:
            try:
                stored_documents = get_documents_by_ids(missing_abstract_ids)
            except Exception as error:
                # Same behaviour as before this lookup existed: score on title/concepts.
                print(
                    "[Reranker] Could not load abstracts for graph-only candidates: "
                    f"{error}"
                )
                stored_documents = {}
            for pid, stored in stored_documents.items():
                if stored.get("document"):
                    vector_map[pid] = stored["document"]
                metadata_map.setdefault(pid, stored.get("metadata") or {})

        candidate_payloads = []
        pairs_to_score = []

        for pid in fused_ids:
            graph_meta = expanded_graph.get(pid, {})
            stored_meta = metadata_map.get(pid, {})
            # Neo4j is the primary source for titles; ChromaDB metadata fills gaps.
            title = graph_meta.get("title") or stored_meta.get("title") or ""
            concepts = ", ".join(graph_meta.get("concepts", []))
            abstract = vector_map.get(pid, "")

            # Construct textual payload for the reranker model
            text_payload = (
                f"Title: {title} | Concepts: {concepts} | Abstract: {abstract}"
            )
            candidate_payloads.append(
                {
                    "paper_id": pid,
                    "title": title,
                    "authors": graph_meta.get("authors", []),
                    "concepts": graph_meta.get("concepts", []),
                    "categories": graph_meta.get("categories", []),
                    "published": stored_meta.get("published"),
                    "abstract": abstract,
                    "text_payload": text_payload,
                }
            )
            pairs_to_score.append((user_query, text_payload))

        # Compute neural relevance scores
        scores = self.reranker.predict(pairs_to_score)

        for idx, score in enumerate(scores):
            candidate_payloads[idx]["score"] = float(score)

        # Sort candidates by Cross-Encoder score descending
        candidate_payloads.sort(key=lambda x: x["score"], reverse=True)
        top_reranked = candidate_payloads[:5]

        source_ids = {item["paper_id"] for item in top_reranked}
        print(f"[Reranker] Selected top {len(top_reranked)} highest scoring papers.")

        graph_records = state.get("graph_raw_results", [])
        vector_sources = []
        for item in vector_raw:
            metadata = item.get("metadata") or {}
            vector_sources.append(
                {
                    "paper_id": str(item["paper_id"]),
                    "title": metadata.get("title"),
                    "published": metadata.get("published"),
                    "excerpt": str(item.get("document") or ""),
                }
            )

        retrieval_debug = {
            "graph_queries": state.get("graph_query_attempts", []),
            "graph_result_count": len(graph_records),
            "graph_records": graph_records[:20],
            "graph_results_truncated": len(graph_records) > 20,
            "vector_hit_count": len(vector_raw),
            "vector_sources": vector_sources,
            "fused_candidate_count": len(fused_ids),
            "reranked_papers": [
                {"paper_id": p["paper_id"], "title": p["title"], "score": p["score"]}
                for p in top_reranked
            ],
        }

        return {
            "reranked_evidence": top_reranked,
            "source_ids": source_ids,
            "retrieval": retrieval_debug,
        }

    def _node_synthesis(self, state: ScholarGraphState) -> dict[str, Any]:
        """Node 7: Synthesizes final response grounded exclusively in reranked evidence."""
        user_query = state["user_query"]
        conversation_history = state.get("conversation_history") or []
        evidence = state["reranked_evidence"]
        source_ids = state["source_ids"]
        retrieval = state["retrieval"]

        print("[7. Synthesis] Generating citation-grounded response via Gemini...")
        history_context = json.dumps(conversation_history[-12:], indent=2)

        synthesis_prompt = f"""
        You are ScholarGraph, an expert academic research assistant.
        Write a detailed, well-organized answer to the USER QUERY using ONLY the
        reranked evidence below. Every claim MUST be supported by that evidence and list
        its supporting paper IDs in source_ids. Never hallucinate facts or citations
        outside ALLOWED SOURCE IDS.

        HOW TO WRITE THE ANSWER:
        - Return between {SYNTHESIS_MIN_CLAIMS} and {SYNTHESIS_MAX_CLAIMS} claims.
        - Claim 1 is an overview: 2-3 sentences that directly answer the question.
        - Each later claim covers one theme, method, or finding in 2-4 sentences: what
          the work does, how it does it (models, methods, data), and what it shows or
          why it matters. Use the specific details in each paper's title, concepts and
          abstract rather than general statements.
        - Connect papers where they agree, differ, or build on each other. Do not just
          restate one abstract per claim.
        - If the evidence only partly answers the question, say what is missing.
        - Put paper IDs ONLY in source_ids. Never write IDs, brackets or citation
          markers inside a statement; citations are added automatically.

        CONVERSATION HISTORY (for context only):
        {history_context}

        USER QUERY: {user_query}

        ALLOWED SOURCE IDS:
        {json.dumps(sorted(source_ids))}

        FINAL RERANKED EVIDENCE PAYLOADS:
        {json.dumps(evidence, indent=2)}

        Return claims in the required grounded JSON format.
        """

        try:
            response = generate_content_with_retry(
                self.client,
                model=state.get(
                    "model_name", load_gemini_model_catalog()["default_model"]
                ),
                contents=synthesis_prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": GroundedResponse,
                },
            )
        except Exception as error:
            raise ModelUnavailableError(
                "Gemini is unavailable while synthesizing response.",
                retrieval=retrieval,
            ) from error

        try:
            grounded_response = GroundedResponse.model_validate_json(response.text)
        except Exception as error:
            raise GroundingValidationError(
                "Model failed to return valid evidence-backed claims."
            ) from error

        for claim in grounded_response.claims:
            unsupported = set(claim.source_ids) - source_ids
            if unsupported:
                raise GroundingValidationError(
                    f"Model returned invalid source citations: {unsupported}"
                )

        agent_result = AgentResult(
            answer=format_grounded_answer(grounded_response.claims, evidence),
            retrieval=retrieval,
        )
        return {"result": agent_result}

    # -----------------------------------------------------------------
    # Public Execution Interfaces
    # -----------------------------------------------------------------

    def run_with_evidence(
        self,
        user_query: str,
        conversation_history: list[dict[str, str]] | None = None,
        model_name: str | None = None,
    ) -> AgentResult:
        """Executes the complete GraphRAG pipeline."""
        if model_name is None:
            model_name = load_gemini_model_catalog()["default_model"]
        initial_state: ScholarGraphState = {
            "user_query": user_query,
            "model_name": model_name,
            "conversation_history": conversation_history or [],
        }
        final_state = self.app.invoke(initial_state)
        return final_state["result"]

    def run(
        self,
        user_query: str,
        conversation_history: list[dict[str, str]] | None = None,
        model_name: str | None = None,
    ) -> str:
        """Backwards compatible string output wrapper."""
        return self.run_with_evidence(
            user_query, conversation_history, model_name
        ).answer
