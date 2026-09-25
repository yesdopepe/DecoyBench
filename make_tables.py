import json
import os

from evaluator import PROMPTS, summarize

MODELS = [
    ("Gemini 3.5 Flash-Lite", "gemini-3.5-flash-lite"),
    ("Gemini 3.6 Flash", "gemini-3.6-flash"),
    ("GPT-5.6 Luna", "gpt-5.6-luna"),
    ("GPT-5.6 Terra", "gpt-5.6-terra"),
    ("Claude Haiku 4.5", "claude-haiku-4-5-20251001"),
    ("Claude Sonnet 5", "claude-sonnet-5"),
]
RESOLUTIONS = [("512x512", ""), ("64x64", "_64x64")]


def load(model_id, suffix):
    with open(os.path.join("results", model_id + suffix, "benchmark_results.json"), encoding="utf-8") as f:
        return json.load(f)["results"]


def main():
    scores = {
        (model_id, res, mode): summarize(load(model_id, suffix), mode)
        for _, model_id in MODELS
        for res, suffix in RESOLUTIONS
        for mode in PROMPTS
    }

    print("Table 3: contour / shading EM (%) and LS")
    print(f"{'Model':22} {'Prompt':7}" + "".join(f" | {res:^24}" for res, _ in RESOLUTIONS))
    for name, model_id in MODELS:
        for mode in PROMPTS:
            row = f"{name:22} {mode.title():7}"
            for res, _ in RESOLUTIONS:
                s = scores[model_id, res, mode]
                row += (f" | {s['contour_em']:5.1f} {s['contour_ls']:.3f}"
                        f"  {s['shading_em']:5.1f} {s['shading_ls']:.3f}")
            print(row)

    print("\nTable 4: average number of extracted texts")
    print(f"{'Model':22} {'Prompt':7}" + "".join(f" {res:>8}" for res, _ in RESOLUTIONS))
    for name, model_id in MODELS:
        for mode in PROMPTS:
            print(f"{name:22} {mode.title():7}"
                  + "".join(f" {scores[model_id, res, mode]['avg_texts']:8.2f}" for res, _ in RESOLUTIONS))


if __name__ == "__main__":
    main()
