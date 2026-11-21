# G2: chia community (Leiden phân cấp) trên G1, tạo node community (H). Không gọi LLM; G3 sẽ tóm tắt.
#   python -m hybrid_graphrag.knowledge_base.graph.build_g2 medical --sweep --stability
#   python -m hybrid_graphrag.knowledge_base.graph.build_g2 medical --resolution 1.0
#   python -m hybrid_graphrag.knowledge_base.graph.build_g2 novel --source Novel-2544 --sweep --hub-pct 0.5
#   python -m hybrid_graphrag.knowledge_base.graph.build_g2 novel --hub-pct 0.5
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


# ---------- nạp G1 và dựng đồ thị đưa vào Leiden ----------
def load_g1(base: Path, suffix: str) -> tuple[nx.Graph, Path]:
    fs = sorted(glob.glob(str(base / f"g1_graph_*-{suffix}.gml")))
    if not fs:
        raise FileNotFoundError(f"Không có g1_graph_*-{suffix}.gml trong {base}")
    return nx.read_gml(fs[0]), Path(fs[0])


def hub_threshold(g: nx.Graph, max_degree: int, hub_pct: float | None, hub_floor: int) -> int:
    """Có hub_pct: ngưỡng = bậc tại phân vị (100 - hub_pct) của entity, không thấp hơn hub_floor."""
    if hub_pct is None:
        return max_degree
    degs = sorted(g.degree(n) for n, d in g.nodes(data=True) if d.get("kind") == "entity")
    if not degs:
        return max_degree
    k = min(len(degs) - 1, int(len(degs) * (1 - hub_pct / 100)))
    return max(degs[k], hub_floor)


def leiden_input(g: nx.Graph, thr: int):
    """unit + entity, bỏ chunk và entity hub. Trọng số cạnh lấy từ G1 (mặc định 1)."""
    hubs = {n for n, d in g.nodes(data=True) if d.get("kind") == "entity" and g.degree(n) > thr}
    keep = [n for n, d in g.nodes(data=True) if d.get("kind") in ("unit", "entity") and n not in hubs]
    sub = g.subgraph(keep)
    nodes = list(sub.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    edges, w = [], []
    for a, b, d in sub.edges(data=True):
        edges.append((idx[a], idx[b]))
        w.append(float(d.get("weight", 1)))
    G = ig.Graph(n=len(nodes), edges=edges, directed=False)
    G.es["weight"] = w
    return sub, nodes, G, hubs


def run_leiden(G: ig.Graph, resolution: float, seed: int) -> list[int]:
    return leidenalg.find_partition(
        G, leidenalg.RBConfigurationVertexPartition,
        weights="weight", resolution_parameter=resolution, seed=seed).membership


# ---------- gộp cụm nhỏ vào cụm lân cận (thay cho giải thể) ----------
def merge_small(G: ig.Graph, kinds: list[str], mem: list[int], min_units: int, caps: tuple[int, ...]) -> list[int]:
    """Cụm có < min_units unit được gộp vào cụm kề có modularity gain lớn nhất,
    với tổng unit <= cap (thử lần lượt từng cap). Cụm không có láng giềng hợp lệ thì giữ nguyên."""
    mem = list(mem)
    el = G.get_edgelist()
    w = G.es["weight"] if el else []
    deg = G.strength(weights="weight") if el else [0.0] * G.vcount()
    m2 = 2 * sum(w) or 1.0

    units, strength = Counter(), Counter()
    for i, c in enumerate(mem):
        strength[c] += deg[i]
        units[c] += kinds[i] == "unit"
    adj = defaultdict(lambda: defaultdict(float))
    for (a, b), x in zip(el, w):
        ca, cb = mem[a], mem[b]
        if ca != cb:
            adj[ca][cb] += x
            adj[cb][ca] += x
    remap: dict[int, int] = {}

    def absorb(c, d):
        units[d] += units.pop(c)
        strength[d] += strength.pop(c)
        for x, wx in list(adj.get(c, {}).items()):
            adj[x].pop(c, None)
            if x != d:
                adj[d][x] += wx
                adj[x][d] += wx
        adj[d].pop(c, None)
        adj.pop(c, None)
        remap[c] = d

    for cap in caps:
        changed = True
        while changed:
            changed = False
            for c in sorted((c for c in units if units[c] < min_units), key=units.get):
                if c not in units or units[c] >= min_units:
                    continue
                cands = [(adj[c][d] - strength[c] * strength[d] / m2, d)
                         for d in adj.get(c, {}) if units[c] + units[d] <= cap]
                if cands:
                    absorb(c, max(cands)[1])
                    changed = True

    def find(c):
        while c in remap:
            c = remap[c]
        return c
    return [find(m) for m in mem]


# ---------- chia đệ quy cụm lớn ----------
def split_large(G: ig.Graph, kinds: list[str], mem: list[int], a, seed: int) -> list[int]:
    """Cụm > max_units unit: chạy Leiden trên chính nó (resolution riêng cho đồ thị con),
    gộp mảnh nhỏ trong nội bộ, rồi đệ quy với mảnh còn lớn."""
    mem = list(mem)
    nxt = max(mem) + 1 if mem else 0
    members = defaultdict(list)
    for i, m in enumerate(mem):
        members[m].append(i)  # ids tăng dần -> induced_subgraph giữ đúng thứ tự
    n_units = lambda ids: sum(kinds[i] == "unit" for i in ids)
    queue = [(c, 0) for c, ids in members.items() if n_units(ids) > a.max_units]

    while queue:
        c, depth = queue.pop()
        ids = members[c]
        sub = G.induced_subgraph(ids)
        sk = [kinds[i] for i in ids]
        part = None
        for k in range(3):  # không chia được thì tăng resolution
            r = a.split_res * (1.5 ** depth) * (2 ** k)
            p = merge_small(sub, sk, run_leiden(sub, r, seed), a.min_units, (a.max_units,))
            if len(set(p)) > 1:
                part = p
                break
        if part is None:
            continue
        groups = defaultdict(list)
        for i, p in zip(ids, part):
            groups[p].append(i)
        for j, gi in enumerate(groups.values()):
            cid = c if j == 0 else nxt
            if j > 0:
                nxt += 1
            members[cid] = gi
            for i in gi:
                mem[i] = cid
            if n_units(gi) > a.max_units and depth + 1 < a.max_depth:
                queue.append((cid, depth + 1))
    return mem


def partition(G, kinds, res, seed, a):
    coarse = run_leiden(G, res, seed)
    mem = split_large(G, kinds, coarse, a, seed)
    mem = merge_small(G, kinds, mem, a.min_units, (a.max_units, int(a.max_units * 1.5)))
    return coarse, mem


# ---------- xử lý unit mồ côi, chốt nhãn ----------
def attach_orphans(g, nodes, kinds, mem, min_units) -> tuple[int, int]:
    """Unit còn nằm trong cụm < min_units (không gộp được): chuyển sang cụm lớn nhất cùng chunk."""
    units = Counter(m for m, k in zip(mem, kinds) if k == "unit")
    big = {c for c, u in units.items() if u >= min_units}
    by_chunk = defaultdict(Counter)
    for n, m, k in zip(nodes, mem, kinds):
        if k == "unit" and m in big:
            by_chunk[g.nodes[n]["chunk_id"]][m] += 1
    moved = left = 0
    for i, (n, m, k) in enumerate(zip(nodes, mem, kinds)):
        if k == "unit" and m not in big:
            v = by_chunk.get(g.nodes[n]["chunk_id"])
            if v:
                mem[i] = v.most_common(1)[0][0]
                moved += 1
            else:
                left += 1
    return moved, left


def finalize(G, kinds, mem) -> list[int]:
    """Đánh số lại 0..k-1 (lớn nhất trước). Entity trong cụm không còn unit nào được gán theo láng giềng, không có thì -1."""
    units = Counter(m for m, k in zip(mem, kinds) if k == "unit")
    remap = {c: i for i, (c, _) in enumerate(units.most_common())}
    lab = [remap.get(m, -1) for m in mem]
    for i, l in enumerate(lab):
        if l == -1:
            v = Counter(lab[j] for j in G.neighbors(i) if lab[j] >= 0)
            if v:
                lab[i] = v.most_common(1)[0][0]
    return lab


def as_membership(lab: list[int]) -> list[int]:
    k = max(lab) + 1
    return [l if l >= 0 else k + i for i, l in enumerate(lab)]


# ---------- thống kê ----------
def stats(g, G, nodes, kinds, lab, a, moved, left) -> dict:
    units_per, chunks_per = Counter(), defaultdict(set)
    for n, k, l in zip(nodes, kinds, lab):
        if k == "unit":
            units_per[l] += 1
            chunks_per[l].add(g.nodes[n]["chunk_id"])
    sizes = sorted(units_per.values(), reverse=True)
    el = G.get_edgelist()
    inside = sum(1 for x, y in el if lab[x] >= 0 and lab[x] == lab[y])
    return {
        "communities": len(sizes),
        "units_median": statistics.median(sizes) if sizes else 0,
        "units_max": sizes[0] if sizes else 0,
        "units_over_max": sum(s > a.max_units for s in sizes),
        "units_under_min": sum(s < a.min_units for s in sizes),
        "chunks_median": statistics.median(len(v) for v in chunks_per.values()) if chunks_per else 0,
        "intra_edge_pct": 100 * inside / max(len(el), 1),
        "modularity": G.modularity(as_membership(lab), weights="weight") if el else 0.0,
        "orphans_moved": moved,
        "orphans_left": left,
    }


def fmt(res, s, a) -> str:
    return (f"res={res:<4} cụm={s['communities']:>4} | unit/cụm: median {s['units_median']:.0f}, max {s['units_max']}, "
            f">{a.max_units}: {s['units_over_max']}, <{a.min_units}: {s['units_under_min']} | "
            f"chunk median {s['chunks_median']:.0f} | mồ côi chuyển {s['orphans_moved']}, còn {s['orphans_left']} | "
            f"cạnh nội cụm {s['intra_edge_pct']:.0f}% | modularity (cuối) {s['modularity']:.3f}")


# ---------- ghi node community ----------
def add_communities(g: nx.Graph, assign: dict[str, int], parent: dict[int, int]) -> tuple[nx.Graph, list[dict]]:
    g2 = g.copy()
    for n, d in g2.nodes(data=True):
        if d["kind"] in ("unit", "entity"):
            d["community"] = assign.get(n, -1)  # -1: hub hoặc entity không gán được
    members = defaultdict(lambda: {"units": [], "entities": []})
    for n, c in assign.items():
        members[c]["units" if g.nodes[n]["kind"] == "unit" else "entities"].append(n)

    info = []
    for c in sorted(members):
        units, ents = members[c]["units"], members[c]["entities"]
        top = [e[2:] for e in sorted(ents, key=g.degree, reverse=True)[:10]]
        g2.add_node(f"H:{c}", kind="community", community=c, parent=parent.get(c, -1),
                    n_units=len(units), n_entities=len(ents), top_entities=" | ".join(top))
        for u in units:
            g2.add_edge(f"H:{c}", u, kind="summarizes")
        info.append({"community": c, "parent": parent.get(c, -1), "units": units, "entities": ents,
                     "top_entities": top, "chunk_ids": sorted({g.nodes[u]["chunk_id"] for u in units})})
    return g2, info


def process(domain: str, source: str | None, a) -> None:
    base = index_dir(domain, source if domain == "novel" else None)
    g, gpath = load_g1(base, a.suffix)
    thr = hub_threshold(g, a.max_degree, a.hub_pct, a.hub_floor)
    sub, nodes, G, hubs = leiden_input(g, thr)
    kinds = [sub.nodes[n]["kind"] for n in nodes]
    print(f"\n===== {source or domain} | {gpath.name} =====")
    print(f"Đưa vào Leiden: {len(nodes)} node, {G.ecount()} cạnh; bỏ {len(hubs)} hub (bậc > {thr})")

    for res in ([0.5, 1.0, 2.0] if a.sweep else [a.resolution]):
        coarse, mem = partition(G, kinds, res, a.seed, a)
        moved, left = attach_orphans(g, nodes, kinds, mem, a.min_units)
        lab = finalize(G, kinds, mem)
        s = stats(g, G, nodes, kinds, lab, a, moved, left)
        print(fmt(res, s, a))
        if a.stability:
            nmis = []
            for k in (1, 2):
                _, m2 = partition(G, kinds, res, a.seed + k, a)
                attach_orphans(g, nodes, kinds, m2, a.min_units)
                nmis.append(ig.compare_communities(as_membership(lab), as_membership(finalize(G, kinds, m2)), method="nmi"))
            print(f"  [ổn định] NMI trung bình giữa các seed = {statistics.mean(nmis):.3f}")
        if a.sweep:
            continue

        assign = {n: l for n, l in zip(nodes, lab) if l >= 0}
        votes = defaultdict(Counter)  # parent của cụm cuối = cụm tầng 1 có nhiều unit nhất
        for k, l, cc in zip(kinds, lab, coarse):
            if k == "unit" and l >= 0:
                votes[l][cc] += 1
        parent = {l: v.most_common(1)[0][0] for l, v in votes.items()}

        g2, info = add_communities(g, assign, parent)
        tag = gpath.stem.removeprefix("g1_graph_")
        nx.write_gml(g2, str(base / f"g2_graph_{tag}.gml"))
        (base / f"g2_communities_{tag}.json").write_text(
            json.dumps({"resolution": res, "split_res": a.split_res, "hub_threshold": thr,
                        "min_units": a.min_units, "max_units": a.max_units, "max_depth": a.max_depth,
                        "stats": s, "hubs": sorted(h[2:] for h in hubs), "communities": info},
                       ensure_ascii=False), encoding="utf-8")
        print(f"[g2] đã lưu g2_graph_{tag}.gml và g2_communities_{tag}.json "
              f"(top hub bị loại: {[h[2:] for h in sorted(hubs, key=g.degree, reverse=True)[:5]]})")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("domain", choices=["medical", "novel"])
    ap.add_argument("--source", help="novel: một cuốn (mặc định: cả 20)")
    ap.add_argument("--suffix", default="full")
    ap.add_argument("--resolution", type=float, default=1.0, help="resolution Leiden tầng 1")
    ap.add_argument("--split-res", type=float, default=1.0, help="resolution khởi đầu khi chia cụm lớn")
    ap.add_argument("--max-degree", type=int, default=100, help="entity bậc lớn hơn mức này bị bỏ (khi không dùng --hub-pct)")
    ap.add_argument("--hub-pct", type=float, default=None, help="bỏ top x%% entity theo bậc (vd 0.5)")
    ap.add_argument("--hub-floor", type=int, default=20, help="sàn ngưỡng khi dùng --hub-pct")
    ap.add_argument("--min-units", type=int, default=5, help="cụm ít unit hơn mức này được gộp vào cụm kề")
    ap.add_argument("--max-units", type=int, default=30, help="cụm nhiều unit hơn mức này bị chia tiếp")
    ap.add_argument("--max-depth", type=int, default=4, help="số tầng chia đệ quy tối đa")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stability", action="store_true", help="đo NMI giữa 3 seed trên kết quả cuối")
    ap.add_argument("--sweep", action="store_true", help="thử res 0.5/1/2, chỉ in thống kê, không lưu")
    a = ap.parse_args()
    targets = [None] if a.domain == "medical" else (
        [a.source] if a.source else [d["corpus_name"] for d in load_corpus("novel")])
    for s in targets:
        process(a.domain, s, a)


if __name__ == "__main__":
    main()