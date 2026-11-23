# G3b: gắn tóm tắt vào node H, embed unit + H, thêm cạnh ngữ nghĩa. Không gọi LLM.
# Chọn cạnh: top-k (bỏ cặp cùng chunk TRƯỚC khi lấy k), cosine >= floor, và (mutual top-k hoặc cosine >= strong).
#   python -m hybrid_graphrag.knowledge_base.graph.build_g3 medical --sweep
#   python -m hybrid_graphrag.knowledge_base.graph.build_g3 novel --source Novel-44557 --sweep
#   python -m hybrid_graphrag.knowledge_base.graph.build_g3 medical
#   python -m hybrid_graphrag.knowledge_base.graph.build_g3 novel --k 5 --floor 0.5 --strong 0.7
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

EXTRA = 16  # số ứng viên dư để còn đủ k sau khi loại cặp cùng chunk


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


def collect_nodes(g: nx.Graph):
    ids, texts = [], []
    for n, d in g.nodes(data=True):
        if d.get("kind") == "unit" and d.get("text"):
            ids.append(n); texts.append(d["text"])
        elif d.get("kind") == "community" and d.get("description"):
            ids.append(n); texts.append(d["description"])
    return ids, texts


def get_embeddings(base: Path, tag: str, ids: list[str], texts: list[str], recompute: bool) -> np.ndarray:
    ef, nf = base / f"g3_embeddings_{tag}.npy", base / f"g3_nodes_{tag}.json"
    if not recompute and ef.exists() and nf.exists():
        if json.loads(nf.read_text(encoding="utf-8")) == ids:
            print("  dùng embedding đã lưu (thêm --recompute để tính lại)")
            return np.load(ef)
    return embed_texts(texts, show_progress=False)


def candidates(emb: np.ndarray, kmax: int):
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    return index.search(emb, min(kmax + EXTRA + 1, len(emb)))


def select_pairs(chunk: list, sims, nbrs, k: int, floor: float, strong: float) -> dict:
    """Mỗi node lấy k láng giềng hợp lệ (không self, không cùng chunk); giữ cặp nếu cosine >= floor
    và (mutual hoặc cosine >= strong)."""
    top = []
    for i in range(len(chunk)):
        valid = {}
        for s, j in zip(sims[i], nbrs[i]):
            j = int(j)
            if j == i or j < 0:
                continue
            if chunk[i] is not None and chunk[i] == chunk[j]:
                continue
            valid[j] = float(s)
            if len(valid) == k:
                break
        top.append(valid)
    pairs = {}
    for i, nb in enumerate(top):
        for j, s in nb.items():
            if s >= floor and (i in top[j] or s >= strong):
                pairs[(min(i, j), max(i, j))] = s
    return pairs


def describe(pairs: dict, n: int) -> str:
    deg = np.zeros(n, dtype=int)
    for i, j in pairs:
        deg[i] += 1; deg[j] += 1
    return f"{len(pairs)} cặp, {2 * len(pairs) / max(n, 1):.2f} cạnh/node, {int((deg == 0).sum())} node không có cạnh"


def process(domain: str, source: str | None, a) -> None:
    base = index_dir(domain, source if domain == "novel" else None)
    g, tag = load_g2(base, a.suffix)
    n_desc = attach_summaries(g, base / f"g3_summaries_{tag}.jsonl")
    n_h = sum(1 for _, d in g.nodes(data=True) if d.get("kind") == "community")
    print(f"\n===== {source or domain} | H có tóm tắt: {n_desc}/{n_h} =====")
    if n_desc < n_h:
        print(f"  CẢNH BÁO: thiếu {n_h - n_desc} tóm tắt; node H đó không được embed")
    ids, texts = collect_nodes(g)
    emb = np.ascontiguousarray(get_embeddings(base, tag, ids, texts, a.recompute), dtype="float32")
    faiss.normalize_L2(emb)  # đảm bảo inner product = cosine
    print(f"  embed {len(ids)} node (unit + H), dim={emb.shape[1]}")

    chunk = [g.nodes[n]["chunk_id"] if g.nodes[n].get("kind") == "unit" else None for n in ids]
    kmax = max(a.k, 8) if a.sweep else a.k
    sims, nbrs = candidates(emb, kmax)

    if a.sweep:
        for k in (3, 5, 8):
            for floor, strong in ((0.5, 0.7), (0.5, 0.8), (0.6, 0.8)):
                p = select_pairs(chunk, sims, nbrs, k, floor, strong)
                print(f"  k={k} floor={floor} strong={strong}: {describe(p, len(ids))}")
        return

    pairs = select_pairs(chunk, sims, nbrs, a.k, a.floor, a.strong)
    added = reinforced = skipped = 0
    for (i, j), s in pairs.items():
        u, v = ids[i], ids[j]
        if g.has_edge(u, v):
            if g[u][v].get("kind") == "summarizes":
                skipped += 1  # H-unit cùng cụm: không đổi weight
            else:
                g[u][v]["weight"] = float(g[u][v].get("weight", 1)) + 1.0
                reinforced += 1
            continue
        g.add_edge(u, v, kind="semantic", weight=1.0, similarity=round(s, 4))
        added += 1
    sv = list(pairs.values())
    print(f"  k={a.k} floor={a.floor} strong={a.strong}: {describe(pairs, len(ids))}")
    print(f"  thêm {added} cạnh semantic, tăng weight {reinforced} cạnh có sẵn, bỏ {skipped} cạnh H-unit; "
          f"cosine TB {np.mean(sv) if sv else 0:.3f}")

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
    ap.add_argument("--k", type=int, default=5, help="số láng giềng hợp lệ mỗi node")
    ap.add_argument("--floor", type=float, default=0.5, help="cosine tối thiểu (chỉ loại cặp vô nghĩa)")
    ap.add_argument("--strong", type=float, default=0.7, help="cosine từ mức này thì giữ dù không mutual")
    ap.add_argument("--recompute", action="store_true", help="bỏ cache, embed lại")
    ap.add_argument("--sweep", action="store_true", help="in số cạnh theo nhiều cấu hình, không lưu")
    a = ap.parse_args()
    targets = [None] if a.domain == "medical" else (
        [a.source] if a.source else [d["corpus_name"] for d in load_corpus("novel")])
    for s in targets:
        process(a.domain, s, a)


if __name__ == "__main__":
    main()