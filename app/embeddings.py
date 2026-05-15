"""Embedding backend factory — switch between OpenAI and HuggingFace via env."""

from langchain_core.embeddings import Embeddings

from app.config import EMBEDDING_MODEL, EMBEDDING_PROVIDER


def get_embeddings() -> Embeddings:
    if EMBEDDING_PROVIDER == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    if EMBEDDING_PROVIDER == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=EMBEDDING_MODEL)

    raise ValueError(
        f"Unknown EMBEDDING_PROVIDER={EMBEDDING_PROVIDER!r}. "
        "Use 'openai' or 'huggingface'."
    )
