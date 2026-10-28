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
MAX_WAIT = 120             # mọi key phải chờ lâu hơn mức này (thường là hết quota ngày) -> dừng
DAILY_COOLDOWN = 3600      # quota ngày: kiểm tra lại sau 1 giờ
AUTH_COOLDOWN = 24 * 3600  # key sai/bị từ chối: loại khỏi vòng xoay
EMPTY_LIMIT = 3            # số lần thử khi phản hồi rỗng/không hợp lệ (lý do không rõ)
BLOCKED_LIMIT = 2          # số lần thử khi Gemini báo chặn rõ ràng
BLOCK_MARKERS = ("SAFETY", "PROHIBITED", "BLOCKLIST", "SPII", "RECITATION", "IMAGE_SAFETY")

# Model đã probe: chấp nhận thinking_budget=0. Model khác (3.5-flash-lite, flash-lite-latest...)
# báo 400 nếu gửi tham số này, nên với chúng ta không gửi tham số thinking.
THINKING_OFF_OK = ("gemini-2.5-flash", "gemini-3.1-flash-lite")

SAFETY_SETTINGS = [
    types.SafetySetting(category=c, threshold=types.HarmBlockThreshold.BLOCK_NONE)
    for c in (
        types.HarmCategory.HARM_CATEGORY_HARASSMENT,
        types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
    )
]

_clients: dict[int, genai.Client] = {}
_cooldown: dict[int, float] = {}   # chỉ số key -> thời điểm dùng lại được
_state = {"idx": 0, "last_call": 0.0}
_usage = {"calls": 0, "prompt_tokens": 0, "output_tokens": 0}  # chỉ đếm lượt gọi thật (không tính cache)


class LLMError(RuntimeError):
    pass


class LLMBlockedError(LLMError):
    """Gemini chặn câu này (an toàn, nội dung bị cấm...): bỏ qua câu đó, chạy tiếp."""


class LLMQuotaError(LLMError):
    """Mọi key đều bị giới hạn lâu (thường hết quota ngày): dừng, chạy lại sau."""


def get_usage() -> dict:
    """Số lượt gọi và token thật đã dùng trong phiên chạy này (dùng để đo chi phí G1)."""
    return dict(_usage)


def _log(msg: str) -> None:
    print(f"[llm] {msg}", file=sys.stderr)


def _thinking_cfg(model: str):
    """Cấu hình tắt thinking nếu model cho phép; ngược lại None (không gửi tham số, tránh lỗi 400)."""
    if "pro" in model:
        return None
    if model in THINKING_OFF_OK or model.startswith("gemini-2.5"):
        return types.ThinkingConfig(thinking_budget=0)
    return None


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
        raise LLMQuotaError(
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


def _block_reason(resp) -> str | None:
    """Lý do bị chặn nếu Gemini báo rõ (prompt bị chặn, hoặc finish_reason thuộc nhóm an toàn); ngược lại None."""
    pf = getattr(resp, "prompt_feedback", None)
    br = getattr(pf, "block_reason", None) if pf else None
    if br:
        return f"prompt bị chặn ({br})"
    cands = getattr(resp, "candidates", None)
    if cands:
        fr = str(getattr(cands[0], "finish_reason", "") or "")
        if any(m in fr.upper() for m in BLOCK_MARKERS):
            return f"finish_reason={fr}"
    return None


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
        safety_settings=SAFETY_SETTINGS,
    )
    if response_schema is not None:
        cfg["response_mime_type"] = "application/json"
        cfg["response_schema"] = response_schema
    tc = _thinking_cfg(model)  # tắt thinking nếu model cho phép, để token không bị "ăn" vào max_output_tokens
    if tc is not None:
        cfg["thinking_config"] = tc
    config = types.GenerateContentConfig(**cfg)

    empty, blocked, last_err = 0, 0, ""
    for attempt in range(LLM_MAX_RETRIES):
        idx = _pick_key()
        wait = LLM_MIN_INTERVAL - (time.time() - _state["last_call"])
        if wait > 0:
            time.sleep(wait)
        _state["last_call"] = time.time()
        try:
            resp = _client(idx).models.generate_content(model=model, contents=user_prompt, config=config)
            um = getattr(resp, "usage_metadata", None)
            _usage["calls"] += 1
            _usage["prompt_tokens"] += (getattr(um, "prompt_token_count", 0) or 0) if um else 0
            _usage["output_tokens"] += (getattr(um, "candidates_token_count", 0) or 0) if um else 0

            text = (resp.text or "").strip()
            if text and (validate is None or validate(text)):
                if use_cache:
                    CACHE_DIR.mkdir(parents=True, exist_ok=True)
                    cache.write_text(json.dumps({"text": text}, ensure_ascii=False), encoding="utf-8")
                return text

            reason = _block_reason(resp)
            if reason:
                blocked += 1
                last_err = f"bị chặn: {reason}"
                if blocked >= BLOCKED_LIMIT:
                    raise LLMBlockedError(last_err)
            else:
                empty += 1
                cands = getattr(resp, "candidates", None)
                fr = getattr(cands[0], "finish_reason", None) if cands else "không có candidate"
                last_err = f"phản hồi rỗng hoặc không hợp lệ (finish_reason={fr})"
                if empty >= EMPTY_LIMIT:
                    raise LLMError(f"{last_err}. Có thể max_output_tokens={max_output_tokens} quá nhỏ.")
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
                raise LLMError(f"Lỗi không thể thử lại ({code}): {str(e)[:600]}") from e
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