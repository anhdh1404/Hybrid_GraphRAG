# tests/list_models.py
from google import genai
from hybrid_graphrag.config import GEMINI_API_KEYS

client = genai.Client(api_key=GEMINI_API_KEYS[0])
for m in client.models.list():
    if "generateContent" in (m.supported_actions or []):
        print(m.name)