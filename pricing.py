PRICES_PER_MILLION = {
    "gpt-5.6-terra": {"input": 2.50, "cached_input": 1.25, "output": 10.00},
    "gpt-5.6-luna": {"input": 0.15, "cached_input": 0.075, "output": 0.60},
    "gemini-3.6-flash": {"input": 1.50, "cached_input": 0.375, "output": 9.00},
    "gemini-3.5-flash-lite": {"input": 0.50, "cached_input": 0.125, "output": 3.00},
    "claude-sonnet-5": {"input": 3.00, "cached_input": 0.30, "output": 15.00},
    "claude-haiku-4-5-20251001": {"input": 1.00, "cached_input": 0.10, "output": 5.00},
}


def token_cost(model, prompt_tokens, completion_tokens, cached_tokens=0):
    price = PRICES_PER_MILLION.get(model)
    if price is None:
        return 0.0
    cached = min(cached_tokens, prompt_tokens)
    total = (
        (prompt_tokens - cached) * price["input"]
        + cached * price["cached_input"]
        + completion_tokens * price["output"]
    )
    return round(total / 1_000_000, 6)
