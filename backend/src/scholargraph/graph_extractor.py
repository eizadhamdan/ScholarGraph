import argparse
import json
from google import genai
from pydantic import BaseModel, Field
from tqdm import tqdm
from .config import GEMINI_API_KEY


class DeepConcepts(BaseModel):
    methods_and_models: list[str] = Field(
        description="Key AI methods, algorithms, or architectures used in the abstract"
    )


def extract_graph_triples(
    input_path: str, output_path: str, llm_sample_limit: int = 200
):
    """Extracts structural and semantic entity triples from raw paper JSON."""

    with open(input_path, "r", encoding="utf-8") as f:
        papers = json.load(f)

    # Initialize Gemini client if key exists
    client = None
    if GEMINI_API_KEY:
        try:
            client = genai.Client(api_key=GEMINI_API_KEY)
        except Exception as e:
            print(f"[Warning] Could not initialize Gemini Client: {e}")

    graph_data = []
    print(f"Generating graph triples for {len(papers)} papers...")

    for i, paper in enumerate(tqdm(papers)):
        paper_id = paper["id"]
        title = paper["title"]
        authors = paper.get("authors", [])
        categories = paper.get("categories", [])
        summary = paper.get("summary", "")

        relationships = []

        # 1. Author -> AUTHORED -> Paper
        for author in authors:
            relationships.append(
                {"subject": author, "predicate": "AUTHORED", "object": paper_id}
            )

        # 2. Paper -> IN_CATEGORY -> Category
        for cat in categories:
            relationships.append(
                {"subject": paper_id, "predicate": "IN_CATEGORY", "object": cat}
            )

        # 3. Use OpenAlex Concepts as baseline concepts
        concepts = list(categories)

        # 4. LLM Concept Extraction via Gemini (Enrichment)
        if client and i < llm_sample_limit and summary:
            prompt = f"Title: {title}\nAbstract: {summary}\nExtract 3-5 core AI techniques/methods/architectures."
            try:
                response = client.models.generate_content(
                    model="gemini-3.6-flash",
                    contents=prompt,
                    config={
                        "response_mime_type": "application/json",
                        "response_schema": DeepConcepts,
                    },
                )
                extracted = json.loads(response.text)
                llm_concepts = extracted.get("methods_and_models", [])

                # Combine LLM concepts with OpenAlex categories
                concepts = list(set(concepts + llm_concepts))
            except Exception as e:
                # Print error on first failure to debug API issue
                if i == 0:
                    print(f"\n[Gemini API Error] Paper {paper_id}: {e}")

        # Add Concept Relationships
        for concept in concepts:
            relationships.append(
                {
                    "subject": paper_id,
                    "predicate": "USES_METHOD",
                    "object": concept,
                }
            )

        graph_data.append(
            {
                "paper_id": paper_id,
                "title": title,
                "authors": authors,
                "categories": categories,
                "concepts": concepts,
                "relationships": relationships,
            }
        )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(graph_data, f, indent=2)

    print(f"\nSuccessfully saved updated graph triples to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract knowledge graph triples.")
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw_arxiv_papers.json",
        help="Input JSON path",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/graph_triples.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--llm-limit",
        type=int,
        default=200,
        help="Limit LLM extraction calls",
    )
    args = parser.parse_args()
    extract_graph_triples(args.input, args.output, args.llm_limit)
