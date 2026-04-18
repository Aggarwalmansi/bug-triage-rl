import json
import os
from typing import Dict

try:
    from groq import Groq
except Exception:
    Groq = None


DEFAULT_GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


class GroqJSONClient:
    def __init__(self, model: str = DEFAULT_GROQ_MODEL):
        self.model = model
        api_key = os.environ.get("GROQ_API_KEY")
        self.client = Groq(api_key=api_key) if Groq is not None and api_key else None

    def complete_json(self, prompt: str, fallback: Dict) -> Dict:
        if self.client is None or not os.environ.get("GROQ_API_KEY"):
            return fallback
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content or ""
            return parse_json(content, fallback)
        except Exception:
            return fallback


def parse_json(text: str, fallback: Dict) -> Dict:
    try:
        return json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except Exception:
                return fallback
        return fallback
