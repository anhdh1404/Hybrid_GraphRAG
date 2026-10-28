"""Gọi model mã nguồn mở chạy trên server vLLM (API tương thích OpenAI). Dùng cho G1 khi không đủ quota Gemini."""
import json
import re
import threading

import httpx

from hybrid_graphrag.config import VLLM_MODEL, VLLM_URL
from hybrid_graphrag.generation.llm_client import LLMError

ATTEMPT_TEMPS = (0.0, 0.4, 0.7)  # lần đầu tất định; thử lại thì tăng nhiệt độ để không lặp đúng lỗi cũ
_THINK = re.compile(r"<think>.*?</think>", re.S)
_lock = threading.Lock()
_usage = {"calls": 0, "prompt_tokens": 0, "output_tokens": 0}
_http = httpx.Client(timeout=httpx.Timeout(900.0, connect=10.0), limits=httpx.Limits(max_connections=128))


class VLLMServerError(LLMError):
    """Server vLLM không phản hồi hoặc báo lỗi: dừng lại."""


class VLLMOutputError(LLMError):
    """Server chạy bình thường nhưng đầu ra không hợp lệ sau nhiều lần thử: bỏ qua chunk này."""


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
    max_output_tokens: int = 3000,
    model: str | None = None,
    enable_thinking: bool = False,
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
            "chat_template_kwargs": {"enable_thinking": enable_thinking},
        }
        try:
            r = _http.post(f"{VLLM_URL}/chat/completions", json=body)
        except httpx.HTTPError as e:
            raise VLLMServerError(f"Không kết nối được server vLLM ({VLLM_URL}): {e!r}") from e
        if r.status_code != 200:
            raise VLLMServerError(f"Server vLLM trả {r.status_code}: {r.text[:300]}")

        data = r.json()
        use = data.get("usage") or {}
        with _lock:
            _usage["calls"] += 1
            _usage["prompt_tokens"] += use.get("prompt_tokens", 0) or 0
            _usage["output_tokens"] += use.get("completion_tokens", 0) or 0

        choice = data["choices"][0]
        text = _THINK.sub("", choice["message"].get("content") or "").strip()
        try:
            schema.model_validate_json(text)
            return json.loads(text)
        except ValueError:  # pydantic ValidationError và JSONDecodeError đều kế thừa ValueError
            last = f"finish_reason={choice.get('finish_reason')}, {len(text)} ký tự, đầu ra: {text[:120]!r}"
    raise VLLMOutputError(f"Đầu ra không hợp lệ sau {len(ATTEMPT_TEMPS)} lần thử ({last})")