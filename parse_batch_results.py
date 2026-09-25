import argparse
import json
import os

from evaluator import EMPTY_USAGE, PROMPTS, get_model_dir, load_dataset, print_summary, save_results, score
from pricing import token_cost


def read_record(record):
    if "result" in record:
        result = record["result"]
        if result.get("type") != "succeeded":
            return record["custom_id"], "anthropic", f"[ERROR]: {result.get('error', result.get('type'))}", 0, 0, 0
        message = result["message"]
        text = "".join(b.get("text", "") for b in message["content"] if b.get("type") == "text").strip()
        usage = message.get("usage", {})
        return (record["custom_id"], "anthropic", text, usage.get("input_tokens") or 0,
                usage.get("output_tokens") or 0, usage.get("cache_read_input_tokens") or 0)

    if "key" in record:
        response = record.get("response", {})
        if record.get("error") or "candidates" not in response:
            return record["key"], "google", f"[ERROR]: {record.get('error') or response}", 0, 0, 0
        parts = response["candidates"][0].get("content", {}).get("parts", [])
        text = "".join(p["text"] for p in parts if "text" in p).strip()
        usage = response.get("usageMetadata", {})
        completion = (usage.get("candidatesTokenCount") or 0) + (usage.get("thoughtsTokenCount") or 0)
        return (record["key"], "google", text, usage.get("promptTokenCount") or 0,
                completion, usage.get("cachedContentTokenCount") or 0)

    if record.get("error"):
        return record["custom_id"], "openai", f"[ERROR]: {record['error']}", 0, 0, 0
    body = record["response"]["body"]
    message = body["choices"][0]["message"]
    text = (message.get("content") or message.get("reasoning") or "").strip()
    usage = body.get("usage", {})
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    return (record["custom_id"], "openai", text, usage.get("prompt_tokens") or 0,
            usage.get("completion_tokens") or 0, cached)


def parse_batch_output(batch_file, data_file, output_file, model):
    items = {i["image"]: i for i in load_dataset(data_file)}
    results = {}
    provider = ""

    with open(batch_file, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            custom_id, provider, text, prompt, completion, cached = read_record(json.loads(line))
            stem, mode = custom_id.rsplit("_", 1)
            item = items.get(stem + ".png")
            if item is None or mode not in PROMPTS:
                print(f"skipping unknown id {custom_id}")
                continue

            usage = dict(EMPTY_USAGE) if text.startswith("[ERROR]") else {
                "prompt_tokens": prompt,
                "cached_tokens": cached,
                "completion_tokens": completion,
                "total_tokens": prompt + completion,
                "cost_usd": token_cost(model, prompt, completion, cached),
            }
            result = results.setdefault(item["image"], {k: item[k] for k in ("image", "contour", "shading")})
            result[mode] = score(item["contour"], item["shading"], text, usage)

    results = list(results.values())
    save_results(output_file, model, f"{provider}_batch", results)
    print(f"parsed {len(results)} images from {batch_file} -> {output_file}")
    print_summary(results, model, f"{provider} batch")


def main():
    parser = argparse.ArgumentParser(description="Score a downloaded batch results file.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--input-jsonl")
    parser.add_argument("--data-file", default="decoybench.json")
    parser.add_argument("--output-json")
    args = parser.parse_args()

    out_dir = args.output_dir or get_model_dir(args.model)
    parse_batch_output(
        args.input_jsonl or os.path.join(out_dir, "batch_results.jsonl"),
        args.data_file,
        args.output_json or os.path.join(out_dir, "benchmark_results.json"),
        args.model,
    )


if __name__ == "__main__":
    main()
