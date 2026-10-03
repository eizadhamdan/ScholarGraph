# ChromaDB interface & operations
import argparse
from functools import lru_cache
import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer
from scholargraph.config import CHROMA_PERSIST_DIR, EMBEDDING_MODEL_NAME


def get_chroma_client():
    return chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)


@lru_cache(maxsize=1)
def get_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def import_parquet_to_chroma(parquet_path: str):
    """Hydrates persistent ChromaDB collection from Parquet embeddings."""
    client = get_chroma_client()
    collection = client.get_or_create_collection(name="arxiv_papers")

    print(f"Reading {parquet_path}...")
    df = pd.read_parquet(parquet_path)

    batch_size = 500
    total = len(df)

    print(f"Upserting {total} vectors into ChromaDB at {CHROMA_PERSIST_DIR}...")
    for i in range(0, total, batch_size):
        batch = df.iloc[i : i + batch_size]
        ids = batch["id"].astype(str).tolist()
        documents = batch["summary"].tolist()
        embeddings = batch["embedding"].tolist()
        metadatas = [
            {"title": str(row["title"]), "published": str(row["published"])}
            for _, row in batch.iterrows()
        ]

        collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    print("ChromaDB hydration complete!")


def query_vector_store(query_text: str, n_results: int = 5) -> list[dict]:
    """Performs semantic similarity search over ChromaDB abstracts."""
    client = get_chroma_client()
    collection = client.get_collection(name="arxiv_papers")

    query_embedding = (
        get_embedding_model().encode([query_text], normalize_embeddings=True).tolist()
    )

    results = collection.query(query_embeddings=query_embedding, n_results=n_results)

    formatted = []
    if results and "ids" in results and results["ids"]:
        for i in range(len(results["ids"][0])):
            formatted.append(
                {
                    "paper_id": results["ids"][0][i],
                    "document": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                }
            )
    return formatted


def get_documents_by_ids(paper_ids: list[str]) -> dict[str, dict]:
    """Loads stored abstracts and metadata for specific paper IDs from ChromaDB.

    Used for candidates that came from the graph only and therefore have no
    vector-search hit. IDs that are not in the collection are simply omitted.
    """
    if not paper_ids:
        return {}

    client = get_chroma_client()
    collection = client.get_collection(name="arxiv_papers")
    results = collection.get(
        ids=[str(paper_id) for paper_id in paper_ids],
        include=["documents", "metadatas"],
    )

    ids = results.get("ids") or []
    documents = results.get("documents") or []
    metadatas = results.get("metadatas") or []

    found: dict[str, dict] = {}
    for index, paper_id in enumerate(ids):
        found[str(paper_id)] = {
            "document": (documents[index] if index < len(documents) else None) or "",
            "metadata": (metadatas[index] if index < len(metadatas) else None) or {},
        }
    return found


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ChromaDB management.")
    parser.add_argument(
        "--import",
        dest="import_path",
        type=str,
        help="Path to Parquet file to import",
    )
    args = parser.parse_args()
    if args.import_path:
        import_parquet_to_chroma(args.import_path)
