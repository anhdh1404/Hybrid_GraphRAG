"""Gọi Qwen3-4B-Instruct-2507 chạy trên server vLLM (API tương thích OpenAI). Dùng cho G1 và tóm tắt G3."""
import itertools
import json
import threading

import httpx

from hybrid_graphrag.config import G1_MAX_TOKENS, VLLM_MODEL, VLLM_TIMEOUT, VLLM_URLS
from hybrid_graphrag.generation.llm_client import LLMError

ATTEMPT_TEMPS = (0.0, 0.4, 0.7)  # lần đầu tất định; thử lại thì tăng nhiệt độ để không lặp đúng lỗi cũ
SUMMARY_MAX_TOKENS = 400         # tóm tắt community tối đa ~200 từ
_lock = threading.Lock()
_usage = {"calls": 0, "prompt_tokens": 0, "output_tokens": 0}
_urls = itertools.cycle(VLLM_URLS)   # luân phiên giữa các server (mỗi GPU một server)
_http = httpx.Client(timeout=httpx.Timeout(VLLM_TIMEOUT, connect=10.0), limits=httpx.Limits(max_connections=128))


class VLLMServerError(LLMError):
    """Mất kết nối hoặc server báo lỗi 5xx: dừng lại, chạy lại sau khi sửa server."""


class VLLMOutputError(LLMError):
    """Server chạy bình thường nhưng yêu cầu này lỗi (4xx, quá thời gian chờ hoặc đầu ra không hợp lệ): bỏ qua."""


def get_vllm_usage() -> dict:
    with _lock:
        return dict(_usage)


def vllm_server_ready() -> bool:
    try:
        return all(_http.get(f"{u}/models", timeout=5).status_code == 200 for u in VLLM_URLS)
    except httpx.HTTPError:
        return False


def _next_url() -> str:
    with _lock:
        return next(_urls)


def _post(url: str, body: dict) -> dict:
    try:
        r = _http.post(f"{url}/chat/completions", json=body)
    except httpx.ReadTimeout as e:
        raise VLLMOutputError(f"Quá {VLLM_TIMEOUT}s không có phản hồi từ {url}; bỏ qua, sẽ thử lại ở lần chạy sau") from e
    except httpx.HTTPError as e:
        raise VLLMServerError(f"Không kết nối được server vLLM ({url}): {e!r}") from e
    if r.status_code >= 500:
        raise VLLMServerError(f"Server vLLM ({url}) trả {r.status_code}: {r.text[:300]}")
    if r.status_code != 200:
        raise VLLMOutputError(f"Server vLLM ({url}) từ chối yêu cầu ({r.status_code}): {r.text[:300]}")
    data = r.json()
    use = data.get("usage") or {}
    with _lock:
        _usage["calls"] += 1
        _usage["prompt_tokens"] += use.get("prompt_tokens", 0) or 0
        _usage["output_tokens"] += use.get("completion_tokens", 0) or 0
    return data


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
        choice = _post(_next_url(), body)["choices"][0]
        text = (choice["message"].get("content") or "").strip()
        try:
            schema.model_validate_json(text)
            return json.loads(text)
        except ValueError:  # pydantic ValidationError và JSONDecodeError đều kế thừa ValueError
            last = f"finish_reason={choice.get('finish_reason')}, {len(text)} ký tự, đầu ra: {text[:120]!r}"
    raise VLLMOutputError(f"Đầu ra không hợp lệ sau {len(ATTEMPT_TEMPS)} lần thử ({last})")


def call_vllm_text(
    user_prompt: str,
    system_prompt: str = "",
    max_output_tokens: int = SUMMARY_MAX_TOKENS,
    model: str | None = None,
) -> str:
    """Gọi server vLLM, trả về văn bản thường (không ép JSON). Dùng cho tóm tắt."""
    model = model or VLLM_MODEL
    messages = ([{"role": "system", "content": system_prompt}] if system_prompt else []) + [
        {"role": "user", "content": user_prompt}]
    for temp in (0.0, 0.4):
        body = {"model": model, "messages": messages, "temperature": temp, "max_tokens": max_output_tokens}
        text = (_post(_next_url(), body)["choices"][0]["message"].get("content") or "").strip()
        if text:
            return text
    raise VLLMOutputError("Đầu ra rỗng sau 2 lần thử")