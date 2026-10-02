# Local embedding generation (SentenceTransformers)
import argparse
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL_NAME


def generate_embeddings(input_path: str, output_path: str):
    """Generates dense vector embeddings for paper abstracts using SentenceTransformers."""
    print(f"Loading embedding model ({EMBEDDING_MODEL_NAME})...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)

    df = pd.read_json(input_path)
    print(f"Generating embeddings for {len(df)} abstracts on device: {device}...")

    abstracts = df["summary"].tolist()
    embeddings = model.encode(
        abstracts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    df["embedding"] = embeddings.tolist()
    df.to_parquet(output_path, index=False)
    print(f"Successfully saved embeddings to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate dense vector embeddings.")
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw_arxiv_papers.json",
        help="Input JSON path",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/arxiv_vectors.parquet",
        help="Output Parquet path",
    )
    args = parser.parse_args()
    generate_embeddings(args.input, args.output)
