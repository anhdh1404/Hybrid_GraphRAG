"""Gọi Qwen3-4B-Instruct-2507 chạy trên server vLLM (API tương thích OpenAI). Dùng cho G1."""
import json
import threading

import httpx

from hybrid_graphrag.config import G1_MAX_TOKENS, VLLM_MODEL, VLLM_TIMEOUT, VLLM_URL
from hybrid_graphrag.generation.llm_client import LLMError

ATTEMPT_TEMPS = (0.0, 0.4, 0.7)  # lần đầu tất định; thử lại thì tăng nhiệt độ để không lặp đúng lỗi cũ
_lock = threading.Lock()
_usage = {"calls": 0, "prompt_tokens": 0, "output_tokens": 0}
_http = httpx.Client(timeout=httpx.Timeout(VLLM_TIMEOUT, connect=10.0), limits=httpx.Limits(max_connections=128))


class VLLMServerError(LLMError):
    """Mất kết nối hoặc server báo lỗi 5xx: dừng lại, chạy lại sau khi sửa server."""


class VLLMOutputError(LLMError):
    """Server chạy bình thường nhưng chunk này lỗi (4xx hoặc đầu ra không hợp lệ): bỏ qua chunk."""


def get_vllm_usage() -> dict:
    with _lock:
        return dict(_usage)


def vllm_server_ready() -> bool:
    try:
        return _http.get(f"{VLLM_URL}/models", timeout=5).status_code == 200
    except httpx.HTTPError:
        return False


def call_vllm_json(
    user_prompt: str,
    system_prompt: str,
    schema,
    max_output_tokens: int = G1_MAX_TOKENS,
    model: str | None = None,
) -> dict:
    """Gọi server vLLM, ép output JSON theo lược đồ pydantic; trả về dict đã parse."""
    model = model or VLLM_MODEL
    last = "chưa có phản hồi"
    for temp in ATTEMPT_TEMPS:
        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temp,
            "max_tokens": max_output_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": schema.__name__, "schema": schema.model_json_schema(), "strict": True},
            },
        }
        try:
            r = _http.post(f"{VLLM_URL}/chat/completions", json=body)
        except httpx.HTTPError as e:
            raise VLLMServerError(f"Không kết nối được server vLLM ({VLLM_URL}): {e!r}") from e
        if r.status_code >= 500:
            raise VLLMServerError(f"Server vLLM trả {r.status_code}: {r.text[:300]}")
        if r.status_code != 200:
            # 4xx: thường do prompt quá dài hoặc tham số không hợp lệ
            raise VLLMOutputError(f"Server vLLM từ chối yêu cầu ({r.status_code}): {r.text[:300]}")

        data = r.json()
        use = data.get("usage") or {}
        with _lock:
            _usage["calls"] += 1
            _usage["prompt_tokens"] += use.get("prompt_tokens", 0) or 0
            _usage["output_tokens"] += use.get("completion_tokens", 0) or 0

        choice = data["choices"][0]
        text = (choice["message"].get("content") or "").strip()
        try:
            schema.model_validate_json(text)
            return json.loads(text)
        except ValueError:  # pydantic ValidationError và JSONDecodeError đều kế thừa ValueError
            last = f"finish_reason={choice.get('finish_reason')}, {len(text)} ký tự, đầu ra: {text[:120]!r}"
    raise VLLMOutputError(f"Đầu ra không hợp lệ sau {len(ATTEMPT_TEMPS)} lần thử ({last})")