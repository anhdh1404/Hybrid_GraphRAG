# Kiểm tra server vLLM trước khi chạy G1: kết nối, JSON theo lược đồ, tốc độ song song trên chunk thật.
#   python -m tests.probe_vllm            (mặc định 16 yêu cầu song song)
#   python -m tests.probe_vllm 32
import concurrent.futures as cf
import json
import sys
import time

import httpx

from hybrid_graphrag.config import G1_MAX_TOKENS, VLLM_MODEL, VLLM_URL
from hybrid_graphrag.generation.vllm_client import call_vllm_json, get_vllm_usage, vllm_server_ready
from hybrid_graphrag.knowledge_base.graph.build_g1 import load_chunks
from hybrid_graphrag.prompts.extraction import Extraction, build_extraction_prompt, clean_extraction

TOTAL_CHUNKS = 5454
PASSAGE = (
    "In 1962, Maria Lopez, a nurse at St. Anne's Hospital in Lyon, noticed that patients who walked in the "
    "garden each morning recovered faster from heart surgery. She recorded the recovery times of 40 patients "
    "and presented the results to Dr. Henri Dubois, the head of cardiology. Dr. Dubois then introduced a daily "
    "walking program in the hospital, and within two years it was adopted by three other hospitals in France."
)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 16

    print("1) Kết nối")
    if not vllm_server_ready():
        sys.exit(f"KHÔNG kết nối được {VLLM_URL}. Xem vllm.log.")
    models = httpx.get(f"{VLLM_URL}/models", timeout=10).json()["data"]
    running = [m["id"] for m in models]
    print("   model đang chạy:", running, "| cấu hình VLLM_MODEL =", VLLM_MODEL)
    if VLLM_MODEL not in running:
        print("   CẢNH BÁO: VLLM_MODEL không khớp model đang chạy, các lần gọi sẽ lỗi 404.")

    print("\n2) Trả lời thường")
    r = httpx.post(f"{VLLM_URL}/chat/completions", timeout=120, json={
        "model": VLLM_MODEL,
        "messages": [{"role": "user", "content": "Say hi in 3 words."}],
        "max_tokens": 40, "temperature": 0,
    })
    if r.status_code != 200:
        sys.exit(f"   LỖI {r.status_code}: {r.text[:400]}")
    print("   content:", repr((r.json()["choices"][0]["message"].get("content") or "")[:100]))

    print("\n3) JSON theo lược đồ trên một đoạn mẫu")
    system, user = build_extraction_prompt(PASSAGE)
    t0 = time.time()
    data = call_vllm_json(user, system, Extraction, max_output_tokens=G1_MAX_TOKENS)
    print(f"   {time.time() - t0:.1f}s, usage: {get_vllm_usage()}")
    print(json.dumps(clean_extraction(data), ensure_ascii=False, indent=2))

    print(f"\n4) Tốc độ với {n} chunk thật (khác nhau) của medical, chạy song song")
    chunks = load_chunks("medical", None)
    step = max(len(chunks) // n, 1)
    sample = [chunks[i * step] for i in range(n) if i * step < len(chunks)]
    prompts = [build_extraction_prompt(c["text"]) for c in sample]
    before = get_vllm_usage()
    t0 = time.time()
    ok = 0
    with cf.ThreadPoolExecutor(max_workers=n) as ex:
        futs = [ex.submit(call_vllm_json, u, s, Extraction, G1_MAX_TOKENS) for s, u in prompts]
        for f in futs:
            try:
                f.result()
                ok += 1
            except Exception as e:
                print("   lỗi:", str(e)[:200])
    dt = time.time() - t0
    after = get_vllm_usage()
    out_tok = after["output_tokens"] - before["output_tokens"]
    print(f"   {ok}/{len(sample)} chunk xong trong {dt:.1f}s: {out_tok} token ra, {out_tok / dt:.0f} token/giây")
    print(f"   ước lượng G1 cho {TOTAL_CHUNKS:,} chunk: khoảng {TOTAL_CHUNKS * dt / (len(sample) * 3600):.1f} giờ")


if __name__ == "__main__":
    main()