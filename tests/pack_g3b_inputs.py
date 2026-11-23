# python -m tests.pack_g3b_inputs
import zipfile
from pathlib import Path

root = Path(__file__).resolve().parents[1]
out = root / "g3b_inputs.zip"
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for pat in ("g2_graph_*-full.gml", "g3_summaries_*-full.jsonl"):
        for p in (root / "artifacts").rglob(pat):
            z.write(p, p.relative_to(root))
print(out, round(out.stat().st_size / 1e6, 1), "MB")