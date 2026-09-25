import base64
import json
import os
import re

SYSTEM_PROMPT = """You are an OCR expert. Respond ONLY with a valid JSON object. No markdown, no code fences, no commentary.
Schema: {"texts": ["<the full text of one item>"]}
The "texts" array may contain any number of items, including zero.
Rules: 
1. Each item of the "texts" array is the full text you read as one unit, transcribed completely. Do not break one text into grammatical units or single words.

Wrong Output: ["THE TREES", "HAVE", "BLOSSOMED"]
Right Output: ["THE TREES HAVE BLOSSOMED"]

2. Report only what you actually read. Do not guess.


"""

NAIVE_PROMPT = "Transcribe all text phrases written in this image."

GUIDED_PROMPT = """The attached image contains two different texts superimposed on each other. They are not in separate areas of the image: at each letter position, one letter is drawn with thin, sharp contour lines, and another letter is formed by soft, diffuse shading that spreads across and beyond the contour outlines.
Step 1: Read the text formed by the thin contour lines, in full.
Step 2: Then ignore the contour lines entirely. Attend only to the broad pattern of shading across the image, as if the image were blurred and the sharp lines were gone.
Step 3: Read the text formed by that shading, in full.
Return the contour text as the first item of "texts", and the shading text as the second item. Each is a single item containing all of its words. If you read only one text, return only that one item.

"""

PROMPTS = {"naive": NAIVE_PROMPT, "guided": GUIDED_PROMPT}

EMPTY_USAGE = {"prompt_tokens": 0, "cached_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost_usd": 0.0}


def load_dataset(path="decoybench.json", limit=0):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data[:limit] if limit else data


def get_model_dir(model_name, base_dir="results"):
    name = re.sub(r"[^\w.-]", "_", (model_name or "default").strip())
    path = os.path.join(base_dir, name)
    os.makedirs(path, exist_ok=True)
    return path


def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def normalize(text):
    return re.sub(r"\s+", " ", text or "").strip().lower()


def levenshtein(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(prev[j - 1] if ca == cb else 1 + min(prev[j], cur[j - 1], prev[j - 1]))
        prev = cur
    return prev[-1]


def similarity(target, prediction):
    a, b = normalize(target), normalize(prediction)
    if not a and not b:
        return 1.0
    return round(1.0 - levenshtein(a, b) / max(len(a), len(b)), 4)


def exact_match(target, prediction):
    return normalize(target) == normalize(prediction)


def _collect_strings(obj, out):
    if isinstance(obj, str):
        text = re.sub(r"\s+", " ", obj).strip()
        if text:
            out.append(text)
    elif isinstance(obj, list):
        for item in obj:
            _collect_strings(item, out)
    elif isinstance(obj, dict):
        keys = ["texts", "phrases", "predictions", "results", "text1", "text2", "hidden", "decoy", "naive", "guided"]
        known = [k for k in keys if obj.get(k) is not None]
        for key in known or [k for k, v in obj.items() if v is not None]:
            _collect_strings(obj[key], out)


def _json_candidates(text):
    fenced = re.findall(r"```(?:json)?\s*([\[\{].*?[\]\}])\s*```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced
    curly, square = text.find("{"), text.find("[")
    if curly != -1 and (square == -1 or curly < square):
        end = text.rfind("}")
        return [text[curly:end + 1]] if end > curly else []
    if square != -1:
        end = text.rfind("]")
        return [text[square:end + 1]] if end > square else []
    return []


def parse_response(raw):
    if not raw or raw.startswith("[ERROR]"):
        return []

    texts = []
    for block in _json_candidates(raw.strip()):
        try:
            _collect_strings(json.loads(block.strip()), texts)
        except json.JSONDecodeError:
            pass

    if not texts:
        pattern = r'"(?:texts?|text\d*|decoy|hidden|phrase\d*|naive|guided)"\s*:\s*(?:"([^"]+)"|\[(.*?)\])'
        for single, array in re.findall(pattern, raw, re.DOTALL | re.IGNORECASE):
            _collect_strings(single or re.findall(r'"([^"]+)"', array), texts)

    texts = list(dict.fromkeys(texts))
    if len(texts) > 1 and all(len(t.split()) == 1 for t in texts):
        texts.append(" ".join(texts))
    return texts


def match_layers(contour, shading, texts):
    if not texts:
        return None, None
    if len(texts) == 1:
        if similarity(contour, texts[0]) >= similarity(shading, texts[0]):
            return texts[0], None
        return None, texts[0]

    best, pair = -1.0, (0, 1)
    for i, ci in enumerate(texts):
        for j, sj in enumerate(texts):
            if i != j:
                score = similarity(contour, ci) + similarity(shading, sj)
                if score > best:
                    best, pair = score, (i, j)
    return texts[pair[0]], texts[pair[1]]


def score(contour, shading, raw, usage=None):
    texts = parse_response(raw)
    contour_pred, shading_pred = match_layers(contour, shading, texts)
    return {
        "raw_response": raw,
        "texts": texts,
        "num_texts": len(texts),
        "contour_prediction": contour_pred,
        "shading_prediction": shading_pred,
        "contour_em": contour_pred is not None and exact_match(contour, contour_pred),
        "shading_em": shading_pred is not None and exact_match(shading, shading_pred),
        "contour_ls": similarity(contour, contour_pred) if contour_pred is not None else 0.0,
        "shading_ls": similarity(shading, shading_pred) if shading_pred is not None else 0.0,
        "usage": usage or dict(EMPTY_USAGE),
    }


def is_complete(result):
    return all(
        result.get(mode) and not result[mode]["raw_response"].startswith("[ERROR]")
        for mode in PROMPTS
    )


def image_number(result):
    match = re.search(r"\d+", result["image"])
    return int(match.group()) if match else 0


def _usage_total(results, field):
    return sum(r[mode]["usage"].get(field, 0) for r in results for mode in PROMPTS if r.get(mode))


def save_results(path, model, provider, results):
    results = sorted(results, key=image_number)
    cost = _usage_total(results, "cost_usd")
    output = {
        "model": model,
        "provider": provider,
        "sample_count": len(results),
        "usage": {
            "prompt_tokens": _usage_total(results, "prompt_tokens"),
            "cached_tokens": _usage_total(results, "cached_tokens"),
            "completion_tokens": _usage_total(results, "completion_tokens"),
            "cost_usd": round(cost, 6),
        },
        "results": results,
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    os.replace(tmp, path)


def summarize(results, mode):
    n = len(results) or 1
    rows = [r[mode] for r in results if r.get(mode)]
    return {
        "contour_em": 100 * sum(r["contour_em"] for r in rows) / n,
        "contour_ls": sum(r["contour_ls"] for r in rows) / n,
        "shading_em": 100 * sum(r["shading_em"] for r in rows) / n,
        "shading_ls": sum(r["shading_ls"] for r in rows) / n,
        "avg_texts": sum(r["num_texts"] for r in rows) / n,
    }


def print_summary(results, model, provider=""):
    if not results:
        print("No results.")
        return

    prompt = _usage_total(results, "prompt_tokens")
    completion = _usage_total(results, "completion_tokens")
    cost = _usage_total(results, "cost_usd")

    print(f"\n{model} ({provider}) - {len(results)} images")
    print(f"tokens: {prompt:,} in / {completion:,} out, cost ${cost:.4f}")
    print(f"{'':8} {'Contour EM':>10} {'LS':>6} {'Shading EM':>11} {'LS':>6} {'Avg texts':>10}")
    for mode in PROMPTS:
        s = summarize(results, mode)
        print(f"{mode:8} {s['contour_em']:10.1f} {s['contour_ls']:6.3f} "
              f"{s['shading_em']:11.1f} {s['shading_ls']:6.3f} {s['avg_texts']:10.2f}")
    print()
