# python data_statistic\g3_check.py                      (medical + 1 novel, 10 cặp mỗi bên)
# python data_statistic\g3_check.py --source Novel-2544 --pairs 15
# python data_statistic\g3_check.py --all                (chỉ thống kê cả 21 index, không in cặp)
import argparse, glob, json, random, sys
from collections import Counter
from pathlib import Path

import networkx as nx
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1] / "artifacts"


def find(base: Path, pat: str):
    fs = sorted(glob.glob(str(base / pat)))
    return Path(fs[0]) if fs else None


def check(base: Path, name: str, n_pairs: int, seed: int) -> dict:
    gp, ep, np_ = find(base, "g3_graph_*.gml"), find(base, "g3_embeddings_*.npy"), find(base, "g3_nodes_*.json")
    g = nx.read_gml(str(gp))
    emb, ids = np.load(ep), json.loads(np_.read_text(encoding="utf-8"))

    kinds = Counter(d.get("kind") for _, d in g.nodes(data=True))
    ekinds = Counter(d.get("kind", "relation") for _, _, d in g.edges(data=True))
    sem = [(u, v, d) for u, v, d in g.edges(data=True) if d.get("kind") == "semantic"]
    sims = [float(d["similarity"]) for _, _, d in sem]
    h_nodes = [n for n, d in g.nodes(data=True) if d.get("kind") == "community"]
    h_missing = sum(1 for n in h_nodes if not g.nodes[n].get("description"))
    sem_deg = Counter()
    for u, v, _ in sem:
        sem_deg[u] += 1; sem_deg[v] += 1
    embedded_no_sem = sum(1 for n in ids if sem_deg[n] == 0)
    norms = np.linalg.norm(emb, axis=1)

    print(f"\n===== {name} =====")
    print(f"node: {dict(kinds)}")
    print(f"cạnh: {dict(ekinds)}")
    print(f"embedding: {emb.shape}, khớp g3_nodes: {len(ids) == len(emb)}, "
          f"norm TB {norms.mean():.3f} (kỳ vọng ~1.0)")
    print(f"H thiếu description: {h_missing}/{len(h_nodes)}")
    if sims:
        q = np.percentile(sims, [0, 25, 50, 75, 100])
        print(f"cosine semantic: min {q[0]:.3f} | Q1 {q[1]:.3f} | median {q[2]:.3f} | Q3 {q[3]:.3f} | max {q[4]:.3f}")
    print(f"node embed không có cạnh semantic: {embedded_no_sem}/{len(ids)}")

    if n_pairs:
        rnd = random.Random(seed)
        # lấy mẫu phân tầng: nửa vùng thấp (<0.6), nửa còn lại
        low = [e for e in sem if float(e[2]["similarity"]) < 0.6]
        high = [e for e in sem if float(e[2]["similarity"]) >= 0.6]
        pick = rnd.sample(low, min(n_pairs // 2, len(low))) + rnd.sample(high, min(n_pairs - n_pairs // 2, len(high)))
        for u, v, d in pick:
            def show(n):
                x = g.nodes[n]
                t = x.get("text") or x.get("description") or ""
                return f"[{x.get('kind')}] {t[:220]}"
            print(f"\n--- cosine {float(d['similarity']):.3f} ---\nA {show(u)}\nB {show(v)}")
    return {"name": name, "nodes": g.number_of_nodes(), "edges": g.number_of_edges(),
            "semantic": len(sem), "h_missing": h_missing, "no_sem": embedded_no_sem, "embedded": len(ids)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="Novel-2544")
    ap.add_argument("--pairs", type=int, default=10)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    if a.all:
        rows = [check(ROOT / "medical", "medical", 0, a.seed)]
        for d in sorted((ROOT / "novel").iterdir()):
            if d.is_dir():
                rows.append(check(d, d.name, 0, a.seed))
        print("\n| Corpus | Node | Cạnh | Semantic | Node embed | Không cạnh semantic | H thiếu mô tả |\n|---|---|---|---|---|---|---|")
        for r in rows:
            print(f"| {r['name']} | {r['nodes']} | {r['edges']} | {r['semantic']} | {r['embedded']} | "
                  f"{r['no_sem']} ({100 * r['no_sem'] / max(r['embedded'], 1):.0f}%) | {r['h_missing']} |")
        return
    check(ROOT / "medical", "medical", a.pairs, a.seed)
    check(ROOT / "novel" / a.source, a.source, a.pairs, a.seed)


if __name__ == "__main__":
    main()