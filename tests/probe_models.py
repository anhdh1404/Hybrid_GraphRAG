# Kiểm tra server vLLM trước khi chạy G1: kết nối, chế độ suy luận, JSON theo lược đồ, tốc độ song song.
#   python -m tests.probe_vllm            (mặc định 16 yêu cầu song song)
#   python -m tests.probe_vllm 32
import concurrent.futures as cf
import json
import sys
import time

import httpx

from hybrid_graphrag.config import VLLM_MODEL, VLLM_URL
from hybrid_graphrag.generation.vllm_client import call_vllm_json, get_vllm_usage, vllm_server_ready
from hybrid_graphrag.prompts.extraction import Extraction, build_extraction_prompt, clean_extraction

sys.stdout.reconfigure(encoding="utf-8")
PASSAGE = (
    "In 1962, Maria Lopez, a nurse at St. Anne's Hospital in Lyon, noticed that patients who walked in the "
    "garden each morning recovered faster from heart surgery. She recorded the recovery times of 40 patients "
    "and presented the results to Dr. Henri Dubois, the head of cardiology. Dr. Dubois then introduced a daily "
    "walking program in the hospital, and within two years it was adopted by three other hospitals in France."
)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 16

print("1) Kết nối")
if not vllm_server_ready():
    sys.exit(f"KHÔNG kết nối được {VLLM_URL}. Xem vllm.log.")
models = httpx.get(f"{VLLM_URL}/models", timeout=10).json()["data"]
print("   model đang chạy:", [m["id"] for m in models], "| cấu hình VLLM_MODEL =", VLLM_MODEL)

print("\n2) Chế độ suy luận (enable_thinking=false)")
r = httpx.post(f"{VLLM_URL}/chat/completions", timeout=120, json={
    "model": VLLM_MODEL,
    "messages": [{"role": "user", "content": "Say hi in 3 words."}],
    "max_tokens": 80, "temperature": 0,
    "chat_template_kwargs": {"enable_thinking": False},
})
if r.status_code != 200:
    sys.exit(f"   LỖI {r.status_code}: {r.text[:400]}")
msg = r.json()["choices"][0]["message"]
raw = msg.get("content") or ""
print("   content:", repr(raw[:200]))
print("   có thẻ <think> trong content:", "<think>" in raw, "| có reasoning_content:", bool(msg.get("reasoning_content")))

print("\n3) JSON theo lược đồ trên một đoạn mẫu")
system, user = build_extraction_prompt(PASSAGE)
t0 = time.time()
data = call_vllm_json(user, system, Extraction, max_output_tokens=3000)
print(f"   {time.time() - t0:.1f}s, usage: {get_vllm_usage()}")
print(json.dumps(clean_extraction(data), ensure_ascii=False, indent=2))

print(f"\n4) Tốc độ với {N} yêu cầu song song (cùng một đoạn nên prefix cache có thể làm phần đọc prompt nhanh hơn thực tế)")
before = get_vllm_usage()
t0 = time.time()
with cf.ThreadPoolExecutor(max_workers=N) as ex:
    list(ex.map(lambda _: call_vllm_json(user, system, Extraction, max_output_tokens=3000), range(N)))
dt = time.time() - t0
out_tok = get_vllm_usage()["output_tokens"] - before["output_tokens"]
print(f"   {N} yêu cầu xong trong {dt:.1f}s: {out_tok} token ra, {out_tok / dt:.0f} token/giây")
print(f"   ước lượng G1 cho 5.454 chunk: khoảng {5454 * dt / (N * 3600):.1f} giờ (nếu chunk thật có độ dài tương tự đoạn mẫu)")