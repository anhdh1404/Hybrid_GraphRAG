"""
Thống kê G1 (đồ thị entity-relation) cho báo cáo.
Chạy từ thư mục gốc:  python -m data_statistic.g1_stat
                      python -m data_statistic.g1_stat --suffix full --no-plot
Đầu ra: data_statistic/g1_stats.json, g1_stats.md, g1_degree.png
"""
import argparse
import glob
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

import networkx as nx

from hybrid_graphrag.config import ARTIFACTS_DIR
from hybrid_graphrag.data.loader import load_corpus
from hybrid_graphrag.knowledge_base.graph.build_g1 import load_chunks

OUT_DIR = Path(__file__).parent
TOP_HUBS = 10


def find(base: Path, kind: str, suffix: str):
    fs = sorted(glob.glob(str(base / f"g1_{kind}_*-{suffix}.*")))
    return Path(fs[0]) if fs else None


def analyze(name: str, base: Path, n_chunks: int, suffix: str) -> dict | None:
    up, gp = find(base, "units", suffix), find(base, "graph", suffix)
    if not (up and gp):
        return None
    recs = [json.loads(l) for l in up.read_text(encoding="utf-8").splitlines() if l.strip()]
    units = [u for r in recs for u in r["units"]]

    g = nx.read_gml(str(gp))
    ents = [n for n, d in g.nodes(data=True) if d.get("kind") == "entity"]
    rel_edges = [(a, b, d) for a, b, d in g.edges(data=True) if d.get("kind") == "relation"]
    deg = [g.degree(n) for n in ents]
    comps = sorted((len(c) for c in nx.connected_components(g)), reverse=True)
    hubs = sorted(((g.degree(n), n[2:]) for n in ents), reverse=True)[:TOP_HUBS]

    return {
        "name": name,
        "chunks": n_chunks,
        "chunks_covered": len(recs),
        "chunks_no_unit": sum(1 for r in recs if not r["units"]),
        "units": len(units),
        "units_per_chunk": len(units) / max(len(recs), 1),
        "entities_per_unit": sum(len(u["entities"]) for u in units) / max(len(units), 1),
        "relations_per_unit": sum(len(u["relations"]) for u in units) / max(len(units), 1),
        "nodes": g.number_of_nodes(),
        "edges": g.number_of_edges(),
        "entity_nodes": len(ents),
        "relation_edges": len(rel_edges),
        "components": len(comps),
        "largest_component_pct": 100 * comps[0] / g.number_of_nodes(),
        "degree_mean": statistics.mean(deg) if deg else 0,
        "degree_median": statistics.median(deg) if deg else 0,
        "degree_max": max(deg) if deg else 0,
        "entities_degree1_pct": 100 * sum(1 for d in deg if d == 1) / max(len(deg), 1),
        "top_hubs": hubs,
        "_degrees": deg,
    }


def to_markdown(rows: list[dict]) -> str:
    h = ("| Corpus | Chunk | Unit | Entity/unit | Rel/unit | Node | Cạnh | Entity node | "
         "Deg TB | Deg max | Entity deg=1 | TP lớn nhất |\n|---|---|---|---|---|---|---|---|---|---|---|---|\n")
    body = "".join(
        f"| {r['name']} | {r['chunks_covered']}/{r['chunks']} | {r['units']:,} | {r['entities_per_unit']:.1f} | "
        f"{r['relations_per_unit']:.1f} | {r['nodes']:,} | {r['edges']:,} | {r['entity_nodes']:,} | "
        f"{r['degree_mean']:.1f} | {r['degree_max']} | {r['entities_degree1_pct']:.0f}% | "
        f"{r['largest_component_pct']:.1f}% |\n"
        for r in rows
    )
    return h + body


def plot_degrees(rows: list[dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = {"medical": [d for r in rows if r["name"] == "medical" for d in r["_degrees"]],
              "novel (gộp 20 cuốn)": [d for r in rows if r["name"] != "medical" for d in r["_degrees"]]}
    fig, axes = plt.subplots(1, len(groups), figsize=(11, 4))
    for ax, (title, deg) in zip(axes if len(groups) > 1 else [axes], groups.items()):
        if not deg:
            continue
        c = Counter(deg)
        ax.scatter(list(c.keys()), list(c.values()), s=8)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(f"Phân bố bậc entity: {title}")
        ax.set_xlabel("bậc (log)"); ax.set_ylabel("số entity (log)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="full")
    ap.add_argument("--no-plot", action="store_true")
    a = ap.parse_args()

    rows = []
    targets = [("medical", ARTIFACTS_DIR / "medical", len(load_chunks("medical", None)))]
    for d in load_corpus("novel"):
        s = d["corpus_name"]
        targets.append((s, ARTIFACTS_DIR / "novel" / s, len(load_chunks("novel", s))))
    for name, base, n in targets:
        r = analyze(name, base, n, a.suffix)
        if r is None:
            print(f"[bỏ qua] {name}: thiếu file g1_*-{a.suffix}")
            continue
        rows.append(r)
        print(f"{name:<14} {r['chunks_covered']:>4}/{n:<4} unit={r['units']:>5} "
              f"ent/unit={r['entities_per_unit']:.1f} rel/unit={r['relations_per_unit']:.1f} "
              f"node={r['nodes']:>6} cạnh={r['edges']:>6} deg1={r['entities_degree1_pct']:.0f}% "
              f"hub={r['top_hubs'][0][1] if r['top_hubs'] else '-'}")

    nov = [r for r in rows if r["name"] != "medical"]
    if nov:
        tot = lambda k: sum(r[k] for r in nov)
        print(f"\nNOVEL (tổng {len(nov)} cuốn): {tot('chunks_covered')} chunk, {tot('units'):,} unit, "
              f"{tot('nodes'):,} node, {tot('edges'):,} cạnh, {tot('entity_nodes'):,} entity")
    med = next((r for r in rows if r["name"] == "medical"), None)
    if med:
        print("\nTop hub medical:", med["top_hubs"])

    public = [{k: v for k, v in r.items() if k != "_degrees"} for r in rows]
    (OUT_DIR / "g1_stats.json").write_text(json.dumps(public, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "g1_stats.md").write_text(to_markdown(rows), encoding="utf-8")
    print(f"\nĐã lưu: {OUT_DIR / 'g1_stats.json'}, {OUT_DIR / 'g1_stats.md'}")
    if not a.no_plot and rows:
        plot_degrees(rows, OUT_DIR / "g1_degree.png")
        print(f"Đã lưu: {OUT_DIR / 'g1_degree.png'}")


if __name__ == "__main__":
    main()