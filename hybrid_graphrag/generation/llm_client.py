import hashlib
import json
import re
import sys
import time

import httpx
from google import genai
from google.genai import errors, types

from hybrid_graphrag.config import (
    ARTIFACTS_DIR, GEMINI_API_KEYS, LLM_MAX_RETRIES, LLM_MIN_INTERVAL, LLM_MODEL, LLM_TEMPERATURE,
)

CACHE_DIR = ARTIFACTS_DIR / "cache" / "llm"
MAX_WAIT = 120            # mọi key phải chờ lâu hơn mức này (thường là hết quota ngày) -> dừng
DAILY_COOLDOWN = 3600     # quota ngày: kiểm tra lại sau 1 giờ
AUTH_COOLDOWN = 24 * 3600  # key sai/bị từ chối: loại khỏi vòng xoay
EMPTY_LIMIT = 3

_clients: dict[int, genai.Client] = {}
_cooldown: dict[int, float] = {}   # chỉ số key -> thời điểm dùng lại được
_state = {"idx": 0, "last_call": 0.0}


class LLMError(RuntimeError):
    pass


def _log(msg: str) -> None:
    print(f"[llm] {msg}", file=sys.stderr)


def _client(i: int) -> genai.Client:
    if i not in _clients:
        _clients[i] = genai.Client(api_key=GEMINI_API_KEYS[i])
    return _clients[i]


def _pick_key() -> int:
    """Chọn key chưa bị cooldown; nếu tất cả đang chờ thì ngủ đến khi có key, hoặc dừng nếu chờ quá lâu."""
    n = len(GEMINI_API_KEYS)
    now = time.time()
    for off in range(n):
        i = (_state["idx"] + off) % n
        if _cooldown.get(i, 0) <= now:
            _state["idx"] = i
            return i
    i = min(range(n), key=lambda j: _cooldown.get(j, 0))
    wait = _cooldown[i] - now
    if wait > MAX_WAIT:
        raise LLMError(
            f"Mọi key đều đang bị giới hạn (cần chờ ~{wait/60:.0f} phút, có thể đã hết quota ngày). "
            "Chạy lại sau; cache và resume sẽ giữ phần đã làm."
        )
    _log(f"mọi key đang chờ, ngủ {wait:.0f}s")
    time.sleep(wait + 0.5)
    _state["idx"] = i
    return i


def _retry_delay(e: Exception) -> float:
    msg = str(e)
    if "PerDay" in msg or "per day" in msg.lower():
        return DAILY_COOLDOWN
    m = re.search(r"retry in ([\d.]+)s|retryDelay['\"]?:\s*['\"]?([\d.]+)s", msg, re.I)
    return float(m.group(1) or m.group(2)) + 1 if m else 30.0


def _cache_file(*parts):
    h = hashlib.sha256(json.dumps(parts, ensure_ascii=False).encode("utf-8")).hexdigest()
    return CACHE_DIR / f"{h}.json"


def call_gemini(
    user_prompt: str,
    system_prompt: str = "",
    model: str = LLM_MODEL,
    max_output_tokens: int = 1024,
    temperature: float = LLM_TEMPERATURE,
    use_cache: bool = True,
    response_schema=None,   # lớp pydantic -> ép output JSON theo schema
    validate=None,          # hàm str -> bool; chỉ nhận và cache phản hồi hợp lệ
) -> str:
    if not GEMINI_API_KEYS:
        raise LLMError("Không có GEMINI_API_KEY* trong .env / biến môi trường")

    schema_name = getattr(response_schema, "__name__", None)
    cache = _cache_file(model, system_prompt, user_prompt, max_output_tokens, temperature, schema_name)
    if use_cache and cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))["text"]

    cfg = dict(
        system_instruction=system_prompt or None,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
    )
    if response_schema is not None:
        cfg["response_mime_type"] = "application/json"
        cfg["response_schema"] = response_schema
    if "pro" not in model:  # tắt thinking để token không bị "ăn" vào max_output_tokens
        cfg["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
    config = types.GenerateContentConfig(**cfg)

    bad, last_err = 0, ""
    for attempt in range(LLM_MAX_RETRIES):
        idx = _pick_key()
        wait = LLM_MIN_INTERVAL - (time.time() - _state["last_call"])
        if wait > 0:
            time.sleep(wait)
        _state["last_call"] = time.time()
        try:
            resp = _client(idx).models.generate_content(model=model, contents=user_prompt, config=config)
            text = (resp.text or "").strip()
            if text and (validate is None or validate(text)):
                if use_cache:
                    CACHE_DIR.mkdir(parents=True, exist_ok=True)
                    cache.write_text(json.dumps({"text": text}, ensure_ascii=False), encoding="utf-8")
                return text
            bad += 1
            reason = resp.candidates[0].finish_reason if resp.candidates else "không có candidate"
            last_err = f"phản hồi rỗng hoặc không hợp lệ (finish_reason={reason})"
            if bad >= EMPTY_LIMIT:
                raise LLMError(f"{last_err}. Có thể bị chặn an toàn hoặc max_output_tokens={max_output_tokens} quá nhỏ.")
            time.sleep(1)
        except errors.APIError as e:
            code = getattr(e, "code", None)
            last_err = f"{code}: {str(e)[:200]}"
            if code == 429:
                delay = _retry_delay(e)
                _cooldown[idx] = time.time() + delay
                _log(f"key #{idx + 1} bị giới hạn, nghỉ {delay:.0f}s")
            elif code in (401, 403):
                _cooldown[idx] = time.time() + AUTH_COOLDOWN
                _log(f"key #{idx + 1} bị từ chối ({code}), loại khỏi vòng xoay")
            elif code == 404:
                raise LLMError(f"Không tìm thấy model '{model}'. Kiểm tra LLM_MODEL trong settings.py") from e
            elif code and code >= 500:
                time.sleep(min(30, 2 ** attempt))
            else:
                raise LLMError(f"Lỗi không thể thử lại ({last_err})") from e
        except (httpx.HTTPError, OSError) as e:  # lỗi mạng, timeout
            last_err = repr(e)[:200]
            time.sleep(min(30, 2 ** attempt))
    raise LLMError(f"Hết {LLM_MAX_RETRIES} lần thử. Lỗi cuối: {last_err}")


def call_gemini_json(user_prompt: str, system_prompt: str, schema, **kwargs) -> dict:
    """Gọi Gemini với output JSON theo schema pydantic; trả về dict đã parse."""
    def ok(text: str) -> bool:
        try:
            schema.model_validate_json(text)
            return True
        except ValueError:  # pydantic ValidationError kế thừa ValueError
            return False

    text = call_gemini(user_prompt, system_prompt, response_schema=schema, validate=ok, **kwargs)
    return json.loads(text)