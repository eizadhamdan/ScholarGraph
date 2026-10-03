# LangGraph orchestration engine
import json
import re
from dataclasses import dataclass
from typing import Any, TypedDict

from google import genai
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from .config import GEMINI_API_KEY
from .graph_store import (
    GraphQueryError,
    GraphUnavailableError,
    check_graph_connection,
    run_cypher_query,
)
from .vector_store import query_vector_store


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
    statement: str = Field(min_length=1)
    source_ids: list[str] = Field(min_length=1)


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
    conversation_history: list[dict[str, str]]
    cypher_query: str
    graph_results: list[dict[str, Any]]
    graph_queries: list[dict[str, Any]]
    vector_results: list[dict[str, Any]]
    source_ids: set[str]
    retrieval: dict[str, Any]
    result: AgentResult


# =====================================================================
# Constants & Helpers
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
        any(name IN methods + categories WHERE toLower(name) CONTAINS term)
     ] AS matched_terms
WHERE size(matched_terms) > 0
RETURN p.id AS paper_id, p.title AS title, methods, categories, matched_terms
ORDER BY size(matched_terms) DESC, p.id
LIMIT 50
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


def _fallback_terms(user_query: str) -> list[str]:
    normalized = re.sub(r"[^a-z0-9]+", " ", user_query.lower())
    terms = [
        token
        for token in normalized.split()
        if len(token) >= 3 and token not in STOP_WORDS
    ]
    if "retrieval augmented generation" in normalized:
        terms.extend(["retrieval-augmented generation", "rag"])
    return list(dict.fromkeys(terms))[:10]


# =====================================================================
# LangGraph Agent Implementation
# =====================================================================


class ScholarGraphAgent:
    def __init__(self):
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        self.app = self._build_graph()

    def _build_graph(self):
        """Constructs and compiles the LangGraph StateGraph pipeline."""
        workflow = StateGraph(ScholarGraphState)

        # Register Nodes
        workflow.add_node("plan_cypher", self._node_plan_cypher)
        workflow.add_node("retrieve_evidence", self._node_retrieve_evidence)
        workflow.add_node("synthesize_answer", self._node_synthesize_answer)

        # Define Edges
        workflow.add_edge(START, "plan_cypher")
        workflow.add_edge("plan_cypher", "retrieve_evidence")
        workflow.add_edge("retrieve_evidence", "synthesize_answer")
        workflow.add_edge("synthesize_answer", END)

        return workflow.compile()

    # -----------------------------------------------------------------
    # LangGraph Nodes
    # -----------------------------------------------------------------

    def _node_plan_cypher(self, state: ScholarGraphState) -> dict[str, Any]:
        """Node 1: Validates database connectivity and plans Cypher query via LLM."""
        if not check_graph_connection():
            raise GraphUnavailableError(
                "Neo4j is unavailable; refusing to generate a response."
            )

        user_query = state["user_query"]
        print(f"\n[LangGraph: plan_cypher] Analyzing query: '{user_query}'...")

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
        Return only relevant properties and limit results to 50 rows.
        """
        try:
            response = self.client.models.generate_content(
                model="gemini-3.6-flash",
                contents=f"{schema_prompt}\nUser Query: {user_query}",
            )
        except Exception as error:
            raise ModelUnavailableError(
                "Gemini is unavailable while planning the graph query."
            ) from error

        cypher = response.text.replace("```cypher", "").replace("```", "").strip()
        print(f"[Graph Tool] Generated Cypher:\n  {cypher}")
        return {"cypher_query": cypher}

    def _node_retrieve_evidence(self, state: ScholarGraphState) -> dict[str, Any]:
        """Node 2: Fetches evidence from Neo4j (Graph) and ChromaDB (Vector store)."""
        user_query = state["user_query"]
        cypher = state["cypher_query"]

        primary_query_error = None
        try:
            graph_results = run_cypher_query(cypher)
        except GraphQueryError as error:
            graph_results = []
            primary_query_error = str(error)

        print(f"[Graph Tool] Retrieved {len(graph_results)} graph rows.")
        graph_queries = [
            {
                "kind": "Gemini-generated",
                "cypher": cypher,
                "result_count": len(graph_results),
                "error": primary_query_error,
            }
        ]

        # Keyword Fallback search if primary Cypher yielded no rows
        if not graph_results:
            terms = _fallback_terms(user_query)
            if terms:
                fallback_results = run_cypher_query(
                    FALLBACK_GRAPH_QUERY, {"terms": terms}
                )
                graph_queries.append(
                    {
                        "kind": "keyword fallback",
                        "cypher": FALLBACK_GRAPH_QUERY,
                        "parameters": {"terms": terms},
                        "result_count": len(fallback_results),
                    }
                )
                if fallback_results:
                    graph_results = fallback_results
                    print(
                        f"[Graph Tool] Keyword fallback retrieved "
                        f"{len(graph_results)} graph rows."
                    )

        # Vector Store Search
        print("[Vector Tool] Executing semantic search on ChromaDB...")
        vector_results = query_vector_store(user_query, n_results=5)
        print(f"[Vector Tool] Retrieved {len(vector_results)} paper abstracts.")

        # Serialize & Extract allowed Source IDs for grounding verification
        graph_results = json.loads(json.dumps(graph_results, default=str))
        vector_results = json.loads(json.dumps(vector_results, default=str))
        source_ids = {
            str(result["paper_id"])
            for result in vector_results
            if result.get("paper_id") is not None
        }
        for row in graph_results:
            if not isinstance(row, dict):
                continue
            for key, value in row.items():
                normalized_key = key.rsplit(".", 1)[-1].replace("_", " ").lower()
                if (
                    normalized_key in {"id", "paper id", "paperid"}
                    and value is not None
                ):
                    source_ids.add(str(value))

        if not graph_results and not vector_results:
            raise InsufficientEvidenceError(
                "Neither the knowledge graph nor vector database returned evidence."
            )
        if not source_ids:
            raise InsufficientEvidenceError(
                "Retrieved data did not contain source IDs that can be cited."
            )

        retrieval = {
            "graph_queries": graph_queries,
            "graph_result_count": len(graph_results),
            "graph_records": graph_results[:20],
            "graph_results_truncated": len(graph_results) > 20,
            "vector_hit_count": len(vector_results),
            "vector_sources": [
                {
                    "paper_id": str(result["paper_id"]),
                    "title": result.get("metadata", {}).get("title"),
                    "published": result.get("metadata", {}).get("published"),
                    "excerpt": result.get("document", "")[:1200],
                }
                for result in vector_results
                if result.get("paper_id") is not None
            ],
        }

        return {
            "graph_results": graph_results,
            "graph_queries": graph_queries,
            "vector_results": vector_results,
            "source_ids": source_ids,
            "retrieval": retrieval,
        }

    def _node_synthesize_answer(self, state: ScholarGraphState) -> dict[str, Any]:
        """Node 3: Synthesizes structured, grounded answer with mandatory citation verification."""
        user_query = state["user_query"]
        conversation_history = state.get("conversation_history") or []
        source_ids = state["source_ids"]
        graph_results = state["graph_results"]
        vector_results = state["vector_results"]
        retrieval = state["retrieval"]

        print("[LangGraph: synthesize_answer] Synthesizing grounded response...")
        history_context = json.dumps(conversation_history[-12:], indent=2)

        synthesis_prompt = f"""
        You are ScholarGraph, an academic research assistant. Use ONLY the evidence below
        to make factual claims. Conversation history may clarify references but is not evidence.
        Return concise, useful findings. Every finding must cite one or more exact IDs from
        ALLOWED SOURCE IDS. Never invent citations or add facts from your general knowledge.
        Omit claims the retrieved evidence does not support.

        RECENT CONVERSATION (context only, not evidence):
        {history_context}

        USER QUERY: {user_query}

        ALLOWED SOURCE IDS:
        {json.dumps(sorted(source_ids))}

        NEO4J GRAPH RESULTS ({len(graph_results)} rows):
        {json.dumps(graph_results, indent=2)}

        CHROMADB VECTOR RESULTS ({len(vector_results)} papers):
        {json.dumps(vector_results, indent=2)}

        Return the findings in the required structured response. Put the supporting paper IDs
        in source_ids for each finding. Each source ID must appear in ALLOWED SOURCE IDS.
        """

        try:
            response = self.client.models.generate_content(
                model="gemini-3.6-flash",
                contents=synthesis_prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": GroundedResponse,
                },
            )
        except Exception as error:
            raise ModelUnavailableError(
                "Gemini is unavailable while synthesizing the retrieved evidence.",
                retrieval=retrieval,
            ) from error

        try:
            grounded_response = GroundedResponse.model_validate_json(response.text)
        except Exception as error:
            raise GroundingValidationError(
                "Gemini did not return a valid evidence-backed response."
            ) from error

        answer_lines = []
        for claim in grounded_response.claims:
            unsupported_ids = set(claim.source_ids) - source_ids
            if unsupported_ids:
                raise GroundingValidationError(
                    "Gemini returned citations that were not present in retrieved evidence."
                )
            citations = " ".join(f"[{source_id}]" for source_id in claim.source_ids)
            answer_lines.append(f"- {claim.statement} {citations}")

        agent_result = AgentResult(
            answer="\n\n".join(answer_lines), retrieval=retrieval
        )
        return {"result": agent_result}

    # -----------------------------------------------------------------
    # Public Execution Interfaces
    # -----------------------------------------------------------------

    def run_with_evidence(
        self,
        user_query: str,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> AgentResult:
        """Executes the compiled LangGraph workflow state pipeline."""
        initial_state: ScholarGraphState = {
            "user_query": user_query,
            "conversation_history": conversation_history or [],
        }

        # Invoke LangGraph
        final_state = self.app.invoke(initial_state)
        return final_state["result"]

    def run(
        self,
        user_query: str,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> str:
        """Convenience method returning string output for backwards compatibility."""
        return self.run_with_evidence(user_query, conversation_history).answer
