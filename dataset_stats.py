import argparse
import json
from statistics import mean

from evaluator import load_dataset


def jaccard(a, b):
    a, b = set(a.lower().split()), set(b.lower().split())
    return len(a & b) / len(a | b) if a | b else 1.0


def dataset_stats(items):
    words = [w.lower() for i in items for w in (i["contour"] + " " + i["shading"]).split()]
    return {
        "total_text_pairs": len(items),
        "mean_char_length": round(mean(len(i["shading"]) for i in items), 1),
        "mean_word_count": round(mean(len(i["shading"].split()) for i in items), 1),
        "mean_jaccard_similarity": round(mean(jaccard(i["contour"], i["shading"]) for i in items), 2),
        "total_unique_vocabulary": len(set(words)),
        "length_aligned_pairs": sum(
            [len(w) for w in i["contour"].split()] == [len(w) for w in i["shading"].split()] for i in items
        ),
    }


def main():
    parser = argparse.ArgumentParser(description="Summary statistics of the dataset (Table 1).")
    parser.add_argument("--data-file", default="decoybench.json")
    args = parser.parse_args()

    stats = dataset_stats(load_dataset(args.data_file))
    with open("dataset_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
