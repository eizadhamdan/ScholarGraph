# LLM extraction logic
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
    input_path: str, output_path: str, llm_sample_limit: int = 500
):
    """Extracts structural and semantic entity triples from raw paper JSON."""
    client = genai.Client(api_key=GEMINI_API_KEY)

    with open(input_path, "r") as f:
        papers = json.load(f)

    graph_data = []
    print(f"Extracting relationships for {len(papers)} papers...")

    for i, paper in enumerate(tqdm(papers)):
        paper_id = paper["id"]
        title = paper["title"]
        authors = paper.get("authors", [])
        categories = paper.get("categories", [])
        summary = paper.get("summary", "")

        relationships = []

        # Metadata Triples: Author -> AUTHORED -> Paper
        for author in authors:
            relationships.append(
                {"subject": author, "predicate": "AUTHORED", "object": paper_id}
            )

        # Metadata Triples: Paper -> IN_CATEGORY -> Category
        for cat in categories:
            relationships.append(
                {"subject": paper_id, "predicate": "IN_CATEGORY", "object": cat}
            )

        # LLM Concept Extraction for top N abstracts
        concepts = []
        if i < llm_sample_limit and summary:
            prompt = (
                f"Title: {title}\nAbstract: {summary}\nExtract core techniques/models."
            )
            try:
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                    config={
                        "response_mime_type": "application/json",
                        "response_schema": DeepConcepts,
                    },
                )
                extracted = json.loads(response.text)
                concepts = extracted.get("methods_and_models", [])

                for concept in concepts:
                    relationships.append(
                        {
                            "subject": paper_id,
                            "predicate": "USES_METHOD",
                            "object": concept,
                        }
                    )
            except Exception:
                pass

        graph_data.append(
            {
                "paper_id": paper_id,
                "title": title,
                "authors": authors,
                "concepts": concepts,
                "relationships": relationships,
            }
        )

    with open(output_path, "w") as f:
        json.dump(graph_data, f, indent=2)

    print(f"Saved graph triples to {output_path}")


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
        default=500,
        help="Limit LLM extraction calls",
    )
    args = parser.parse_args()
    extract_graph_triples(args.input, args.output, args.llm_limit)
