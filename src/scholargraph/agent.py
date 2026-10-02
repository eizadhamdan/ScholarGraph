# LangGraph orchestration engine
import json
from google import genai
from .config import GEMINI_API_KEY
from .graph_store import run_cypher_query
from .vector_store import query_vector_store


class ScholarGraphAgent:
    def __init__(self):
        self.client = genai.Client(api_key=GEMINI_API_KEY)

    def _generate_cypher(self, user_query: str) -> str:
        """Translates user natural language prompt into a valid Neo4j Cypher query."""
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

        Return ONLY a clean Cypher query without markdown formatting or code blocks.
        """
        response = self.client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=f"{schema_prompt}\nUser Query: {user_query}",
        )
        return response.text.replace("```cypher", "").replace("```", "").strip()

    def run(self, user_query: str) -> str:
        """Agent execution flow routing between Cypher graph traversal, vector search, and answer synthesis."""
        print(f"\n[Agent Thinking] Analyzing query: '{user_query}'...")

        # Step 1: Execute Graph Search
        cypher = self._generate_cypher(user_query)
        print(f"[Graph Tool] Generated Cypher:\n  {cypher}")
        graph_results = run_cypher_query(cypher)

        # Step 2: Execute Vector Search
        print("[Vector Tool] Executing semantic search on ChromaDB...")
        vector_results = query_vector_store(user_query, n_results=3)

        # Step 3: Synthesis Phase
        synthesis_prompt = f"""
        You are ScholarGraph, an expert academic research assistant.
        Answer the user's question using the retrieved context from both our Graph Database (Neo4j) and Vector Database (ChromaDB).

        USER QUERY: {user_query}

        NEO4J GRAPH CONTEXT:
        {json.dumps(graph_results, indent=2)}

        CHROMADB VECTOR CONTEXT:
        {json.dumps(vector_results, indent=2)}

        Synthesize a concise, clear, and well-structured answer with citations where appropriate.
        """

        response = self.client.models.generate_content(
            model="gemini-3.5-flash-lite", contents=synthesis_prompt
        )
        return response.text
