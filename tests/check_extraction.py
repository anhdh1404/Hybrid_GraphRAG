import json
import sys

from hybrid_graphrag.config import index_dir
from hybrid_graphrag.generation.llm_client import call_gemini_json
from hybrid_graphrag.knowledge_base.vector_db import VectorDB
from hybrid_graphrag.prompts.extraction import Extraction, build_extraction_prompt, clean_extraction

sys.stdout.reconfigure(encoding="utf-8")
SOURCE = sys.argv[1] if len(sys.argv) > 1 else "Novel-25646"
IDS = [int(x) for x in sys.argv[2:]] or [21, 64, 76]

db = VectorDB.load(index_dir("novel", SOURCE))
stats: dict = {}

for i in IDS:
    c = db.chunks[i]
    system, user = build_extraction_prompt(c["text"])
    data = call_gemini_json(user, system, Extraction, max_output_tokens=3000)
    units = clean_extraction(data, stats)
    print(f"\n===== {c['chunk_id']}: {len(units)} unit =====\n{c['text'][:300]}...")
    print(json.dumps(units, ensure_ascii=False, indent=2))

print("\nTHỐNG KÊ:", stats)