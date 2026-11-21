import json, glob, os, sys, statistics
sys.stdout.reconfigure(encoding="utf-8")

ROOT = r"G:\Downloads_2\Hybrid_GraphRAG\artifacts"
OUT = r"G:\Downloads_2\Hybrid_GraphRAG\data_statistic"

fs = glob.glob(os.path.join(ROOT, "**", "g2_communities_*full.json"), recursive=True)

def corpus_name(f):
    rel = os.path.relpath(f, ROOT).split(os.sep)
    return rel[-2] if len(rel) >= 2 else rel[0]

rows = []
for f in fs:
    d = json.load(open(f, encoding="utf-8"))
    s = d["stats"]
    rows.append({
        "name": corpus_name(f),
        "res": d.get("resolution"),
        "hub_thr": d.get("hub_threshold"),
        "hubs": len(d["hubs"]),
        "communities": s["communities"],
        "median": s["units_median"],
        "max": s["units_max"],
        "over": s["units_over_max"],
        "under": s["units_under_min"],
        "chunk_median": s["chunks_median"],
        "intra": s["intra_edge_pct"],
        "moved": s["orphans_moved"],
        "left": s["orphans_left"],
        "mod": s["modularity"],
        "units": sum(len(c["units"]) for c in d["communities"]),
        "entities": sum(len(c["entities"]) for c in d["communities"]),
    })

rows.sort(key=lambda r: (r["name"] != "medical", r["name"]))

lines = [
    "| Corpus | Res | Hub thr | Hub bỏ | Cụm | Unit gán | Unit/cụm median | Max | >30 | <min | Chunk median | Cạnh nội cụm | Mồ côi chuyển/còn | Modularity |",
    "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
]
for r in rows:
    lines.append(
        f"| {r['name']} | {r['res']} | {r['hub_thr']} | {r['hubs']} | {r['communities']} | {r['units']} | "
        f"{r['median']:.0f} | {r['max']} | {r['over']} | {r['under']} | {r['chunk_median']:.0f} | "
        f"{r['intra']:.0f}% | {r['moved']}/{r['left']} | {r['mod']:.3f} |")

nov = [r for r in rows if r["name"] != "medical"]
if nov:
    lines += [
        "",
        f"**NOVEL ({len(nov)} cuốn):** tổng {sum(r['communities'] for r in nov)} cụm, "
        f"{sum(r['units'] for r in nov)} unit được gán, "
        f"{sum(r['over'] for r in nov)} cụm >30 unit, "
        f"{sum(r['left'] for r in nov)} unit mồ côi còn lại; "
        f"median số cụm/cuốn = {statistics.median(r['communities'] for r in nov):.0f}, "
        f"median unit/cụm (trung bình các cuốn) = {statistics.mean(r['median'] for r in nov):.1f}, "
        f"cạnh nội cụm trung bình = {statistics.mean(r['intra'] for r in nov):.0f}%",
    ]

text = "\n".join(lines)
print(text)
os.makedirs(OUT, exist_ok=True)
open(os.path.join(OUT, "g2_stats.md"), "w", encoding="utf-8").write(text)
json.dump(rows, open(os.path.join(OUT, "g2_stats.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(f"\nĐã lưu: {OUT}\\g2_stats.md, g2_stats.json")