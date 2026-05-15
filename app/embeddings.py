"""Embedding backend factory — switch between OpenAI and HuggingFace via env."""

from langchain_core.embeddings import Embeddings

from app.config import EMBEDDING_MODEL, EMBEDDING_PROVIDER


def get_embeddings(device: str = "cpu") -> Embeddings:
    """device only applies to HuggingFace ('cpu' or 'cuda'). OpenAI ignores it."""
    if EMBEDDING_PROVIDER == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
        )

    if EMBEDDING_PROVIDER == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=EMBEDDING_MODEL)

    raise ValueError(
        f"Unknown EMBEDDING_PROVIDER={EMBEDDING_PROVIDER!r}. "
        "Use 'openai' or 'huggingface'."
    )


def pick_device() -> str:
    """Return 'cuda' if a CUDA GPU is available, otherwise 'cpu'."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"
