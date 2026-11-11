# G2: chia community (Leiden) trên G1, tạo node community (H). Không gọi LLM; G3 sẽ tóm tắt.
#   python -m hybrid_graphrag.knowledge_base.graph.build_g2 medical --sweep
#   python -m hybrid_graphrag.knowledge_base.graph.build_g2 medical --resolution 1.0
#   python -m hybrid_graphrag.knowledge_base.graph.build_g2 novel                 (cả 20 cuốn)
#   python -m hybrid_graphrag.knowledge_base.graph.build_g2 novel --source Novel-2544 --sweep
import argparse
import glob
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import igraph as ig
import leidenalg
import networkx as nx

from hybrid_graphrag.config import index_dir
from hybrid_graphrag.data.loader import load_corpus


def load_g1(base: Path, suffix: str) -> tuple[nx.Graph, Path]:
    fs = sorted(glob.glob(str(base / f"g1_graph_*-{suffix}.gml")))
    if not fs:
        raise FileNotFoundError(f"Không có g1_graph_*-{suffix}.gml trong {base}")
    return nx.read_gml(fs[0]), Path(fs[0])


def leiden_input(g: nx.Graph, max_degree: int):
    """Đồ thị đưa vào Leiden: unit + entity (bỏ chunk và entity hub). Trọng số cạnh quan hệ = số lần xuất hiện."""
    hubs = {n for n, d in g.nodes(data=True) if d.get("kind") == "entity" and g.degree(n) > max_degree}
    keep = [n for n, d in g.nodes(data=True) if d.get("kind") in ("unit", "entity") and n not in hubs]
    sub = g.subgraph(keep)
    nodes = list(sub.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    edges, w = [], []
    for a, b, d in sub.edges(data=True):
        edges.append((idx[a], idx[b]))
        w.append(float(d.get("weight", 1)))
    ig_g = ig.Graph(n=len(nodes), edges=edges, directed=False)
    ig_g.es["weight"] = w
    return sub, nodes, ig_g, hubs


def run_leiden(ig_g: ig.Graph, resolution: float, seed: int) -> list[int]:
    part = leidenalg.find_partition(
        ig_g, leidenalg.RBConfigurationVertexPartition,
        weights="weight", resolution_parameter=resolution, seed=seed)
    return part.membership


def consolidate(g: nx.Graph, sub: nx.Graph, nodes: list[str], membership: list[int], min_units: int) -> dict[str, int]:
    """Giữ community có >= min_units unit. Unit của community nhỏ chuyển sang community lớn nhất trong cùng chunk."""
    comm = defaultdict(list)
    for n, m in zip(nodes, membership):
        comm[m].append(n)
    is_unit = lambda n: sub.nodes[n]["kind"] == "unit"
    big = {m for m, ns in comm.items() if sum(map(is_unit, ns)) >= min_units}
    assign = {n: m for m in big for n in comm[m]}

    votes = defaultdict(Counter)
    for n, m in assign.items():
        if is_unit(n):
            votes[g.nodes[n]["chunk_id"]][m] += 1
    for m, ns in comm.items():
        if m in big:
            continue
        for n in ns:
            if is_unit(n) and votes.get(g.nodes[n]["chunk_id"]):
                assign[n] = votes[g.nodes[n]["chunk_id"]].most_common(1)[0][0]

    order = [m for m, _ in Counter(assign.values()).most_common()]  # đánh số lại 0..k-1, lớn nhất trước
    remap = {m: i for i, m in enumerate(order)}
    return {n: remap[m] for n, m in assign.items()}


def stats(g: nx.Graph, assign: dict[str, int], n_units_total: int, modularity: float) -> dict:
    units_per, chunks_per = defaultdict(int), defaultdict(set)
    for n, c in assign.items():
        if g.nodes[n]["kind"] == "unit":
            units_per[c] += 1
            chunks_per[c].add(g.nodes[n]["chunk_id"])
    sizes = sorted(units_per.values(), reverse=True)
    return {
        "communities": len(sizes),
        "units_median": statistics.median(sizes) if sizes else 0,
        "units_max": sizes[0] if sizes else 0,
        "units_over_30": sum(1 for s in sizes if s > 30),  # prompt tóm tắt chỉ đọc tối đa 30 unit
        "chunks_median": statistics.median(len(v) for v in chunks_per.values()) if chunks_per else 0,
        "units_assigned_pct": 100 * sum(sizes) / max(n_units_total, 1),
        "modularity": modularity,
    }


def fmt(res, s) -> str:
    return (f"res={res:<5} cộng đồng={s['communities']:>5} | unit/cộng đồng: median {s['units_median']:.0f}, "
            f"max {s['units_max']}, >30: {s['units_over_30']} | chunk median {s['chunks_median']:.0f} | "
            f"unit được gán {s['units_assigned_pct']:.0f}% | modularity {s['modularity']:.3f}")


def add_communities(g: nx.Graph, assign: dict[str, int]) -> tuple[nx.Graph, list[dict]]:
    g2 = g.copy()
    for n, d in g2.nodes(data=True):
        if d["kind"] in ("unit", "entity"):
            d["community"] = assign.get(n, -1)
    members = defaultdict(lambda: {"units": [], "entities": []})
    for n, c in assign.items():
        members[c]["units" if g.nodes[n]["kind"] == "unit" else "entities"].append(n)

    info = []
    for c in sorted(members):
        units, ents = members[c]["units"], members[c]["entities"]
        top = [e[2:] for e in sorted(ents, key=g.degree, reverse=True)[:10]]
        hid = f"H:{c}"
        g2.add_node(hid, kind="community", community=c, n_units=len(units), n_entities=len(ents),
                    top_entities=" | ".join(top))
        for u in units:
            g2.add_edge(hid, u, kind="summarizes")
        info.append({"community": c, "units": units, "entities": ents, "top_entities": top,
                     "chunk_ids": sorted({g.nodes[u]["chunk_id"] for u in units})})
    return g2, info


def process(domain: str, source: str | None, a) -> None:
    base = index_dir(domain, source if domain == "novel" else None)
    g, gpath = load_g1(base, a.suffix)
    n_units = sum(1 for _, d in g.nodes(data=True) if d.get("kind") == "unit")
    sub, nodes, ig_g, hubs = leiden_input(g, a.max_degree)
    print(f"\n===== {source or domain} | {gpath.name} =====")
    print(f"Đưa vào Leiden: {len(nodes)} node, {ig_g.ecount()} cạnh; bỏ {len(hubs)} hub (bậc > {a.max_degree})")

    resolutions = [0.5, 1.0, 2.0, 5.0] if a.sweep else [a.resolution]
    for res in resolutions:
        mem = run_leiden(ig_g, res, a.seed)
        assign = consolidate(g, sub, nodes, mem, a.min_units)
        s = stats(g, assign, n_units, ig_g.modularity(mem, weights="weight"))
        print(fmt(res, s))
        if a.sweep:
            continue
        g2, info = add_communities(g, assign)
        tag = gpath.stem.removeprefix("g1_graph_")
        nx.write_gml(g2, str(base / f"g2_graph_{tag}.gml"))
        (base / f"g2_communities_{tag}.json").write_text(
            json.dumps({"resolution": res, "max_degree": a.max_degree, "min_units": a.min_units,
                        "hubs": sorted(h[2:] for h in hubs), "communities": info}, ensure_ascii=False),
            encoding="utf-8")
        print(f"[g2] đã lưu g2_graph_{tag}.gml và g2_communities_{tag}.json "
              f"(top hub bị loại: {[h[2:] for h in sorted(hubs, key=g.degree, reverse=True)[:5]]})")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("domain", choices=["medical", "novel"])
    ap.add_argument("--source", help="novel: một cuốn (mặc định: cả 20)")
    ap.add_argument("--suffix", default="full")
    ap.add_argument("--resolution", type=float, default=1.0)
    ap.add_argument("--max-degree", type=int, default=100, help="entity bậc lớn hơn mức này bị bỏ khi chia community")
    ap.add_argument("--min-units", type=int, default=3, help="community ít unit hơn mức này bị giải thể")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sweep", action="store_true", help="thử nhiều resolution, chỉ in thống kê, không lưu")
    a = ap.parse_args()
    targets = [None] if a.domain == "medical" else (
        [a.source] if a.source else [d["corpus_name"] for d in load_corpus("novel")])
    for s in targets:
        process(a.domain, s, a)


if __name__ == "__main__":
    main()