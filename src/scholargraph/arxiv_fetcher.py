import json
import pandas as pd
import torch
import time
import requests
from google import genai
from google.colab import userdata, files
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

"""Fetch Papers"""


def reconstruct_abstract(inverted_index):
    """Reconstructs text from OpenAlex's inverted index format."""
    if not inverted_index:
        return ""
    word_list = []
    for word, positions in inverted_index.items():
        for pos in positions:
            word_list.append((pos, word))
    word_list.sort(key=lambda x: x[0])
    return " ".join([w[1] for w in word_list])


TARGET_COUNT = 5000

base_url = "https://api.openalex.org/works"
cursor = "*"
papers = []

# OpenAlex Filter parameters:
# - has_abstract:true -> Guarantees every returned paper has an abstract
# - concepts.id:C41008148|C154945302 -> Computer Science OR Artificial Intelligence
# - from_publication_date:2020-01-01 -> Papers published from 2020 onwards
filter_query = "has_abstract:true,concepts.id:C41008148|C154945302,from_publication_date:2020-01-01"

headers = {
    # Adding a descriptive user-agent is recommended by OpenAlex
    "User-Agent": "ScholarGraph-Research/1.0 (mailto:your_email@example.com)"
}

print(f"Fetching up to {TARGET_COUNT} papers from OpenAlex...")

while len(papers) < TARGET_COUNT and cursor:
    params = {"filter": filter_query, "per_page": 100, "cursor": cursor}

    response = requests.get(base_url, params=params, headers=headers)

    if response.status_code != 200:
        print(f"API Error {response.status_code}: {response.text}")
        break

    data = response.json()
    results = data.get("results", [])

    if not results:
        break

    for item in results:
        abstract = reconstruct_abstract(item.get("abstract_inverted_index", {}))
        if not abstract:
            continue

        authors = [a["author"]["display_name"] for a in item.get("authorships", [])]
        concepts = [c["display_name"] for c in item.get("concepts", [])]

        papers.append(
            {
                "id": item["id"].split("/")[-1],
                "title": item.get("title", ""),
                "authors": authors,
                "categories": concepts,
                "summary": abstract,
                "published": item.get("publication_date"),
            }
        )

        if len(papers) >= TARGET_COUNT:
            break

    # Get next cursor token for pagination
    cursor = data.get("meta", {}).get("next_cursor")
    print(f"Downloaded {len(papers)} / {TARGET_COUNT} papers...")

    time.sleep(0.1)  # Polite API delay

print(f"\nSuccessfully downloaded {len(papers)} papers with full abstracts!")

# Save locally to JSON or Parquet
df = pd.DataFrame(papers)
df.to_json("raw_arxiv_papers.json", orient="records", indent=2)
print("Saved to raw_arxiv_papers.json")

"""Generate Knowledge Graph Triples"""

# 1. Load downloaded papers
with open("raw_arxiv_papers.json", "r") as f:
    papers = json.load(f)

# 2. Setup Gemini Client
client = genai.Client(api_key=userdata.get("GEMINI_API_KEY"))


class DeepConcepts(BaseModel):
    methods_and_models: list[str] = Field(
        description="Key AI methods, architectures, or algorithms used (e.g., Transformer, Mamba, RLHF)"
    )


graph_data = []

print("Processing Graph Relationships...")

# Process metadata relationships for ALL papers + Gemini extraction for first 500 papers
for i, paper in enumerate(tqdm(papers)):
    paper_id = paper["id"]
    title = paper["title"]
    authors = paper["authors"]
    categories = paper["categories"]
    summary = paper["summary"]

    # Structural Triples from Metadata
    relationships = []

    # Author -> AUTHORED -> Paper
    for author in authors:
        relationships.append(
            {"subject": author, "predicate": "AUTHORED", "object": paper_id}
        )

    # Paper -> IN_CATEGORY -> Category
    for cat in categories:
        relationships.append(
            {"subject": paper_id, "predicate": "IN_CATEGORY", "object": cat}
        )

    # Deep Concept Extraction using Gemini (for top 500 abstracts)
    concepts = []
    if i < 500:
        prompt = f"Title: {title}\nAbstract: {summary}\nExtract core techniques/models."
        try:
            response = client.models.generate_content(
                model="gemini-3.5-flash-lite",
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
        except Exception as e:
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

# Save graph payload
with open("graph_triples.json", "w") as f:
    json.dump(graph_data, f, indent=2)

print("Saved graph_triples.json!")

"""Generate Vector Embeddings"""

print("Loading SentenceTransformer model on GPU...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("BAAI/bge-small-en-v1.5", device=device)

# Load raw JSON into pandas
df = pd.DataFrame(papers)

print(f"Generating embeddings for {len(df)} abstracts on {device}...")
abstracts = df["summary"].tolist()

# GPU batch encoding
embeddings = model.encode(
    abstracts, batch_size=64, show_progress_bar=True, normalize_embeddings=True
)

# Attach embeddings and save as compressed Parquet
df["embedding"] = embeddings.tolist()
df.to_parquet("arxiv_vectors.parquet", index=False)

print("Saved arxiv_vectors.parquet successfully!")

"""Download Artifacts"""

print("Downloading database files...")
files.download("raw_arxiv_papers.json")
files.download("graph_triples.json")
files.download("arxiv_vectors.parquet")
