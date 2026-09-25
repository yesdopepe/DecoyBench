import argparse
import json
import os

from evaluator import PROMPTS, SYSTEM_PROMPT, encode_image, get_model_dir, load_dataset
from providers import detect_provider

GEMINI_THINKING_BUDGET = {"low": 1024, "medium": 2048, "high": 4096}


def mime_type(path):
    ext = os.path.splitext(path)[1].lower()
    return {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}.get(ext, "image/png")


def openai_request(custom_id, model, prompt, image_b64, mime, max_tokens, reasoning_effort):
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image_b64}"}},
            ]},
        ],
        "max_completion_tokens": max_tokens,
    }
    if reasoning_effort:
        body["reasoning_effort"] = reasoning_effort
    return {"custom_id": custom_id, "method": "POST", "url": "/v1/chat/completions", "body": body}


def anthropic_request(custom_id, model, prompt, image_b64, mime, max_tokens, reasoning_effort):
    params = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": mime, "data": image_b64}},
            {"type": "text", "text": prompt},
        ]}],
        "system": SYSTEM_PROMPT,
    }
    return {"custom_id": custom_id, "params": params}


def google_request(custom_id, model, prompt, image_b64, mime, max_tokens, reasoning_effort):
    config = {"max_output_tokens": max_tokens}
    if reasoning_effort:
        config["thinking_config"] = {"thinking_budget": GEMINI_THINKING_BUDGET[reasoning_effort]}
    request = {
        "contents": [{"role": "user", "parts": [
            {"text": prompt},
            {"inline_data": {"mime_type": mime, "data": image_b64}},
        ]}],
        "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "generation_config": config,
    }
    return {"key": custom_id, "request": request}


BUILDERS = {"openai": openai_request, "anthropic": anthropic_request, "google": google_request}


def write_requests(items, image_dir, model, provider, output, max_tokens=16384, reasoning_effort=None):
    build = BUILDERS[provider]
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    count = 0
    with open(output, "w", encoding="utf-8") as f:
        for item in items:
            path = os.path.join(image_dir, item["image"])
            if not os.path.exists(path):
                print(f"missing image {path}, skipping")
                continue
            image_b64 = encode_image(path)
            stem = os.path.splitext(item["image"])[0]
            for mode, prompt in PROMPTS.items():
                request = build(f"{stem}_{mode}", model, prompt, image_b64, mime_type(path), max_tokens, reasoning_effort)
                f.write(json.dumps(request) + "\n")
                count += 1
    print(f"wrote {count} requests to {output}")


def main():
    parser = argparse.ArgumentParser(description="Build a batch request file (naive + guided per image).")
    parser.add_argument("--model", required=True)
    parser.add_argument("--provider", choices=list(BUILDERS))
    parser.add_argument("--data-file", default="decoybench.json")
    parser.add_argument("--image-dir", default="images/512x512")
    parser.add_argument("--samples", type=int, default=0, help="0 = all 300 images")
    parser.add_argument("--reasoning-effort", choices=["low", "medium", "high"],
                        help="leave unset to use the provider default")
    parser.add_argument("--output-dir")
    parser.add_argument("--output-jsonl")
    args = parser.parse_args()

    out_dir = args.output_dir or get_model_dir(args.model)
    write_requests(
        load_dataset(args.data_file, args.samples),
        args.image_dir,
        args.model,
        args.provider or detect_provider(args.model),
        args.output_jsonl or os.path.join(out_dir, "batch_requests.jsonl"),
        reasoning_effort=args.reasoning_effort,
    )


if __name__ == "__main__":
    main()
