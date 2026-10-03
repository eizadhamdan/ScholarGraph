# Configuration & environment variables
import json
import os
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

GEMINI_MODELS_FILE = Path(__file__).with_name("gemini_models.json")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-en-v1.5")


class GeminiModel(TypedDict):
    id: str
    display_name: str


class GeminiModelCatalog(TypedDict):
    default_model: str
    models: list[GeminiModel]


def load_gemini_model_catalog() -> GeminiModelCatalog:
    with GEMINI_MODELS_FILE.open(encoding="utf-8") as models_file:
        catalog = json.load(models_file)

    if not isinstance(catalog, dict):
        raise ValueError("Gemini model catalog must be a JSON object.")
    models = catalog.get("models")
    default_model = catalog.get("default_model")
    if not isinstance(models, list) or not models:
        raise ValueError("Gemini model catalog must contain at least one model.")
    if not isinstance(default_model, str) or not default_model:
        raise ValueError("Gemini model catalog must define a default_model.")

    normalized_models: list[GeminiModel] = []
    model_ids: set[str] = set()
    for model in models:
        if (
            not isinstance(model, dict)
            or not isinstance(model.get("id"), str)
            or not model["id"]
            or not isinstance(model.get("display_name"), str)
            or not model["display_name"]
        ):
            raise ValueError(
                "Each Gemini model must define a non-empty id and display_name."
            )
        if model["id"] in model_ids:
            raise ValueError(f"Duplicate Gemini model id: {model['id']}")
        model_ids.add(model["id"])
        normalized_models.append(
            {"id": model["id"], "display_name": model["display_name"]}
        )

    if default_model not in model_ids:
        raise ValueError("The default Gemini model must be included in models.")

    return {"default_model": default_model, "models": normalized_models}
