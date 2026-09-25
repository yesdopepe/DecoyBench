import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from evaluator import (PROMPTS, SYSTEM_PROMPT, encode_image, get_model_dir, is_complete,
                       load_dataset, print_summary, save_results, score)
from providers import ChatClient


def run_item(item, image_dir, client, reasoning_effort):
    image_b64 = encode_image(os.path.join(image_dir, item["image"]))
    result = {"image": item["image"], "contour": item["contour"], "shading": item["shading"]}
    for mode, prompt in PROMPTS.items():
        raw, usage = client.complete(SYSTEM_PROMPT, prompt, image_b64, reasoning_effort=reasoning_effort)
        result[mode] = score(item["contour"], item["shading"], raw, usage)
    return result


def load_previous(path, model):
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if data.get("model") != model:
        return {}
    return {r["image"]: r for r in data["results"] if is_complete(r)}


def main():
    parser = argparse.ArgumentParser(description="Run the benchmark synchronously (no batch API).")
    parser.add_argument("--model", required=True)
    parser.add_argument("--provider", choices=["openai", "google", "anthropic"])
    parser.add_argument("--samples", type=int, default=0, help="0 = all 300 images")
    parser.add_argument("--data-file", default="decoybench.json")
    parser.add_argument("--image-dir", default="images/512x512")
    parser.add_argument("--output")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--reasoning-effort", choices=["low", "medium", "high"],
                        help="leave unset to use the provider default")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    output = args.output or os.path.join(get_model_dir(args.model), "benchmark_results.json")
    client = ChatClient(args.model, args.provider)
    items = [i for i in load_dataset(args.data_file, args.samples)
             if os.path.exists(os.path.join(args.image_dir, i["image"]))]

    done = {} if args.no_resume else load_previous(output, args.model)
    todo = [i for i in items if i["image"] not in done]
    print(f"{args.model}: {len(done)} cached, {len(todo)} to run")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_item, i, args.image_dir, client, args.reasoning_effort) for i in todo]
        for future in as_completed(futures):
            result = future.result()
            done[result["image"]] = result
            save_results(output, args.model, client.provider, list(done.values()))
            print(f"[{len(done)}/{len(items)}] {result['image']}")

    results = list(done.values())
    save_results(output, args.model, client.provider, results)
    print_summary(results, args.model, client.provider)


if __name__ == "__main__":
    main()
