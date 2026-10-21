from functools import lru_cache

import numpy as np

from hybrid_graphrag.config import EMBEDDING_DIM, EMBEDDING_MAX_SEQ_LEN, EMBEDDING_MODEL


@lru_cache(maxsize=1)
def get_model():
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[embed] {EMBEDDING_MODEL} (dim={EMBEDDING_DIM}) on {device}")
    model = SentenceTransformer(EMBEDDING_MODEL, device=device, truncate_dim=EMBEDDING_DIM)
    model.max_seq_length = EMBEDDING_MAX_SEQ_LEN
    return model


def _encode(texts: list[str], prompt_name: str, batch_size: int | None, show_progress: bool):
    model = get_model()
    if batch_size is None:
        batch_size = 64 if str(model.device).startswith("cuda") else 16
    return model.encode(
        texts,
        prompt_name=prompt_name,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
    ).astype("float32")


def embed_texts(
    texts: list[str], batch_size: int | None = None, show_progress: bool = True
) -> np.ndarray:
    """Embed chunk (prompt 'document'). Trả về (n, dim) float32, đã chuẩn hóa L2."""
    if not isinstance(texts, list):
        raise TypeError("texts phải là list[str]; với một câu truy vấn dùng embed_query()")
    return _encode(texts, "document", batch_size, show_progress)


def embed_query(query: str) -> np.ndarray:
    """Embed câu hỏi (prompt 'query'). Trả về (1, dim) float32."""
    if not isinstance(query, str):
        raise TypeError("query phải là str")
    return _encode([query], "query", None, False)

def embed_queries(queries: list[str], show_progress: bool = False) -> np.ndarray:
    """Embed nhiều câu hỏi một lượt (prompt 'query'). Trả về (n, dim) float32."""
    return _encode(queries, "query", None, show_progress)