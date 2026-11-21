# G3b: gắn tóm tắt vào node H, embed unit + H, thêm cạnh ngữ nghĩa (kNN cosine). Không gọi LLM.
#   python -m hybrid_graphrag.knowledge_base.graph.build_g3 medical --sweep
#   python -m hybrid_graphrag.knowledge_base.graph.build_g3 medical --threshold 0.6
#   python -m hybrid_graphrag.knowledge_base.graph.build_g3 novel --threshold 0.6
import argparse
import glob
import json
import sys
from pathlib import Path

import faiss
import networkx as nx
import numpy as np

from hybrid_graphrag.config import index_dir
from hybrid_graphrag.data.loader import load_corpus
from hybrid_graphrag.knowledge_base.embeddings import embed_texts


def load_g2(base: Path, suffix: str):
    fs = sorted(glob.glob(str(base / f"g2_graph_*-{suffix}.gml")))
    if not fs:
        raise FileNotFoundError(f"Không có g2_graph_*-{suffix}.gml trong {base}")
    p = Path(fs[0])
    return nx.read_gml(str(p)), p.stem.removeprefix("g2_graph_")


def attach_summaries(g: nx.Graph, path: Path) -> int:
    n = 0
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                hid = f"H:{r['community']}"
                if hid in g:
                    g.nodes[hid]["description"] = " ".join(r["description"].split())
                    n += 1
    return n


def embed_nodes(g: nx.Graph):
    ids, texts = [], []
    for n, d in g.nodes(data=True):
        if d.get("kind") == "unit" and d.get("text"):
            ids.append(n); texts.append(d["text"])
        elif d.get("kind") == "community" and d.get("description"):
            ids.append(n); texts.append(d["description"])
    return ids, embed_texts(texts, show_progress=False)


def knn_pairs(g: nx.Graph, ids: list[str], emb: np.ndarray, k: int, floor: float):
    """Cặp (i, j, cosine) với i<j, cosine >= floor, bỏ self và hai unit cùng chunk."""
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    sims, nbrs = index.search(emb, min(k + 1, len(ids)))
    pairs = {}
    for i in range(len(ids)):
        for s, j in zip(sims[i], nbrs[i]):
            if j == i or j < 0 or s < floor:
                continue
            a, b = ids[i], ids[j]
            da, db = g.nodes[a], g.nodes[b]
            if da["kind"] == "unit" and db["kind"] == "unit" and da["chunk_id"] == db["chunk_id"]:
                continue
            pairs[(min(i, j), max(i, j))] = float(s)
    return pairs


def process(domain: str, source: str | None, a) -> None:
    base = index_dir(domain, source if domain == "novel" else None)
    g, tag = load_g2(base, a.suffix)
    n_desc = attach_summaries(g, base / f"g3_summaries_{tag}.jsonl")
    n_h = sum(1 for _, d in g.nodes(data=True) if d.get("kind") == "community")
    print(f"\n===== {source or domain} | H có tóm tắt: {n_desc}/{n_h} =====")
    if n_desc < n_h:
        print(f"  CẢNH BÁO: thiếu {n_h - n_desc} tóm tắt; node H đó không được embed")
    ids, emb = embed_nodes(g)
    print(f"  embed {len(ids)} node (unit + H), dim={emb.shape[1]}")

    pairs = knn_pairs(g, ids, emb, a.k, 0.5 if a.sweep else a.threshold)
    if a.sweep:
        for t in (0.5, 0.6, 0.7, 0.8):
            print(f"  ngưỡng {t}: {sum(1 for s in pairs.values() if s >= t)} cạnh ngữ nghĩa")
        return

    added = skipped = 0
    for (i, j), s in pairs.items():
        u, v = ids[i], ids[j]
        if g.has_edge(u, v):
            skipped += 1  # đã có cạnh cấu trúc (vd H-unit cùng cụm): không ghi đè
            continue
        g.add_edge(u, v, kind="semantic", weight=1.0, similarity=round(s, 4))
        added += 1
    sims = [s for s in pairs.values()]
    print(f"  thêm {added} cạnh semantic (bỏ {skipped} cặp đã có cạnh); cosine TB {np.mean(sims):.3f}")

    nx.write_gml(g, str(base / f"g3_graph_{tag}.gml"))
    np.save(base / f"g3_embeddings_{tag}.npy", emb)
    (base / f"g3_nodes_{tag}.json").write_text(json.dumps(ids, ensure_ascii=False), encoding="utf-8")
    print(f"  [g3] đã lưu g3_graph_{tag}.gml, g3_embeddings_{tag}.npy, g3_nodes_{tag}.json")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("domain", choices=["medical", "novel"])
    ap.add_argument("--source")
    ap.add_argument("--suffix", default="full")
    ap.add_argument("--k", type=int, default=8, help="số láng giềng gần nhất mỗi node")
    ap.add_argument("--threshold", type=float, default=0.6, help="cosine tối thiểu để thêm cạnh")
    ap.add_argument("--sweep", action="store_true", help="chỉ in số cạnh theo các ngưỡng, không lưu")
    a = ap.parse_args()
    targets = [None] if a.domain == "medical" else (
        [a.source] if a.source else [d["corpus_name"] for d in load_corpus("novel")])
    for s in targets:
        process(a.domain, s, a)


if __name__ == "__main__":
    main()