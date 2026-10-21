import json
from pathlib import Path

import faiss
import numpy as np


class VectorDB:
    """FAISS IndexFlatIP + danh sách chunk. Một instance = một index (medical, hoặc một cuốn novel)."""

    INDEX_FILE = "vector.faiss"
    CHUNKS_FILE = "chunks.json"   # G1 đọc file này
    META_FILE = "meta.json"

    def __init__(self, index: faiss.Index, chunks: list[dict], meta: dict | None = None):
        if index.ntotal != len(chunks):
            raise ValueError(f"index có {index.ntotal} vector nhưng có {len(chunks)} chunk")
        self.index = index
        self.chunks = chunks
        self.meta = meta or {}
        self.id_to_row = {c["chunk_id"]: i for i, c in enumerate(chunks)}

    @property
    def dim(self) -> int:
        return self.index.d

    # ---------------------------------------------------------------- build
    @classmethod
    def build(cls, embeddings: np.ndarray, chunks: list[dict], meta: dict | None = None) -> "VectorDB":
        if len(embeddings) != len(chunks):
            raise ValueError(f"embeddings ({len(embeddings)}) và chunks ({len(chunks)}) không khớp")
        emb = np.array(embeddings, dtype="float32", copy=True)
        faiss.normalize_L2(emb)  # chuẩn hóa để tích vô hướng = cosine
        index = faiss.IndexFlatIP(emb.shape[1])
        index.add(emb)
        meta = {**(meta or {}), "n_chunks": len(chunks), "dim": int(emb.shape[1])}
        return cls(index, chunks, meta)

    # ----------------------------------------------------------- save / load
    @classmethod
    def exists(cls, out_dir: Path) -> bool:
        return all((out_dir / f).exists() for f in (cls.INDEX_FILE, cls.CHUNKS_FILE, cls.META_FILE))

    def save(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(out_dir / self.INDEX_FILE))
        (out_dir / self.CHUNKS_FILE).write_text(
            json.dumps(self.chunks, ensure_ascii=False), encoding="utf-8")
        (out_dir / self.META_FILE).write_text(
            json.dumps(self.meta, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, out_dir: Path) -> "VectorDB":
        index = faiss.read_index(str(out_dir / cls.INDEX_FILE))
        chunks = json.loads((out_dir / cls.CHUNKS_FILE).read_text(encoding="utf-8"))
        meta = json.loads((out_dir / cls.META_FILE).read_text(encoding="utf-8"))
        return cls(index, chunks, meta)

    @classmethod
    def read_meta(cls, out_dir: Path) -> dict:
        return json.loads((out_dir / cls.META_FILE).read_text(encoding="utf-8"))

    # --------------------------------------------------------------- search
    def search(self, query_vec: np.ndarray, top_k: int) -> list[dict]:
        """[{**chunk, 'vector_score'}, ...] giảm dần. query_vec đã chuẩn hóa (embed_query lo việc này)."""
        q = np.asarray(query_vec, dtype="float32").reshape(1, -1)
        scores, rows = self.index.search(q, min(top_k, self.index.ntotal))
        return [
            {**self.chunks[r], "vector_score": float(s)}
            for s, r in zip(scores[0], rows[0])
            if r != -1
        ]

    # --------- dùng cho hybrid scoring (chunk do graph tìm ra, không nằm trong top_k vector)
    def get_embeddings(self, chunk_ids: list[str]) -> np.ndarray:
        """(len(chunk_ids), dim). Báo KeyError nếu chunk_id không tồn tại."""
        rows = [self.id_to_row[c] for c in chunk_ids]
        return np.vstack([self.index.reconstruct(r) for r in rows]).astype("float32")

    def score(self, query_vec: np.ndarray, chunk_ids: list[str]) -> dict[str, float]:
        """Cosine giữa câu truy vấn và các chunk chỉ định -> {chunk_id: score}."""
        if not chunk_ids:
            return {}
        q = np.asarray(query_vec, dtype="float32").reshape(-1)
        sims = self.get_embeddings(chunk_ids) @ q
        return dict(zip(chunk_ids, map(float, sims)))

    def get_chunk(self, chunk_id: str) -> dict:
        return self.chunks[self.id_to_row[chunk_id]]