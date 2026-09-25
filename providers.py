import os

from dotenv import load_dotenv
from openai import OpenAI

from evaluator import EMPTY_USAGE
from pricing import token_cost

load_dotenv()

ENDPOINTS = {
    "openai": ("https://api.openai.com/v1", ["OPENAI_API_KEY"]),
    "google": ("https://generativelanguage.googleapis.com/v1beta/openai/", ["GEMINI_API_KEY", "GOOGLE_API_KEY"]),
    "anthropic": ("https://api.anthropic.com/v1", ["ANTHROPIC_API_KEY"]),
}


def detect_provider(model):
    if model.startswith(("gpt-", "o1", "o3", "o4")):
        return "openai"
    if model.startswith("gemini-"):
        return "google"
    if model.startswith("claude-"):
        return "anthropic"
    raise ValueError(f"can't tell the provider for '{model}', pass --provider")


class ChatClient:
    def __init__(self, model, provider=None):
        self.model = model
        self.provider = provider or detect_provider(model)
        base_url, key_names = ENDPOINTS[self.provider]
        api_key = next((os.environ[k] for k in key_names if os.environ.get(k)), None)
        if not api_key:
            raise ValueError(f"missing {' or '.join(key_names)} in .env")
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def complete(self, system_prompt, user_prompt, image_b64, max_tokens=16384, reasoning_effort=None):
        kwargs = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                ]},
            ],
        }
        if self.provider == "openai":
            kwargs["max_completion_tokens"] = max_tokens
        else:
            kwargs["max_tokens"] = max_tokens
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort

        try:
            response = self.client.chat.completions.create(**kwargs)
        except Exception as e:
            return f"[ERROR]: {e}", dict(EMPTY_USAGE)

        message = response.choices[0].message
        text = (message.content or getattr(message, "reasoning", "") or "").strip()

        usage = response.usage
        prompt = getattr(usage, "prompt_tokens", 0) or 0
        completion = getattr(usage, "completion_tokens", 0) or 0
        details = getattr(usage, "prompt_tokens_details", None)
        cached = getattr(details, "cached_tokens", 0) or 0

        return text, {
            "prompt_tokens": prompt,
            "cached_tokens": cached,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
            "cost_usd": token_cost(self.model, prompt, completion, cached),
        }
