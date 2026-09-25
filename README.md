# DecoyBench

Code, data and model outputs for the paper
**"Sorry Robot, Happy Human: Vision-Language Models Read Only One of Two Legible Typographic Layers"**
(Mert İncidelen, Yamen Kashkash, Asya Berker, Murat Aydoğan — Fırat University).

<p align="center"><img src="assets/example.png" width="300" alt="DecoyBench example"></p>

Each DecoyBench image contains two texts drawn on top of each other with the
[Decoy Font](https://www.mixfont.com/experiments/decoy-font) method: a **contour** text made of thin, sharp
outlines (above: *SORRY ROBOT*) and a **shading** text made of soft, diffuse shading (above: *HAPPY HUMAN*).
People can read both texts. The six vision-language models we tested read the contour text almost perfectly
at 512×512 but almost never the shading text. At 64×64 the contour lines disappear, and both people and
models read the shading text instead.

## Dataset

`decoybench.json` has 300 entries:

```json
{"image": "out_1.png", "contour": "sorry robot", "shading": "happy human"}
```

In each pair, the contour and shading texts have the same number of words and matching word lengths.
`python dataset_stats.py` reproduces Table 1 of the paper:

| Metric | Value |
|---|---|
| Total text pairs | 300 |
| Mean char length | 16.9 |
| Mean word count | 3.8 |
| Mean Jaccard similarity | 0.06 |
| Total unique vocabulary | 576 |

The images used in the evaluation are included in the repository:

```
images/
  512x512/    used for the high-resolution runs
  64x64/      used for the low-resolution runs
```

The 2400×2400 originals (348 MB) are available as `decoybench-original.zip` under
[Releases](../../releases). Extract the zip into `images/original/`. Both evaluation resolutions were made
from the originals with Lanczos resampling:

```bash
python resize_images.py --dst-dir images/512x512 --size 512
python resize_images.py --dst-dir images/64x64 --size 64
```

## Setup

```bash
pip install -r requirements.txt
```

To run models, create a `.env` file with the keys for the providers you use:

```
OPENAI_API_KEY=...
GEMINI_API_KEY=...
ANTHROPIC_API_KEY=...
```

## Running a model

Every image is sent twice, once with the naive prompt and once with the guided prompt, using a shared system
prompt. The prompts are in `evaluator.py` and in Appendix B of the paper. The paper's runs used the batch APIs:

```bash
python batch_generator.py --model gpt-5.6-luna --image-dir images/512x512
python run_batch_job.py --model gpt-5.6-luna
python parse_batch_results.py --model gpt-5.6-luna
```

The outputs are written to `results/<model>/` (512×512) and `results/<model>_64x64/` (64×64).
For the 64×64 run, use `--image-dir images/64x64 --output-dir results/gpt-5.6-luna_64x64` in the first command
and `--output-dir results/gpt-5.6-luna_64x64` in the other two.

`run_batch_job.py` splits the requests into 6 jobs by default (`--num-chunks`) and waits until they finish.
Use `--action submit` to submit and exit, then `--batch-id <ids> --action download` to download later.
The provider is detected from the model name. To choose it yourself, pass `--provider openai|google|anthropic`.

`benchmark.py` does the same without the batch API. It calls the provider directly and resumes from the
last saved result:

```bash
python benchmark.py --model claude-sonnet-5 --image-dir images/512x512
```

Models evaluated in the paper:

| Family | Models |
|---|---|
| GPT | `gpt-5.6-luna`, `gpt-5.6-terra` |
| Gemini | `gemini-3.6-flash`, `gemini-3.5-flash-lite` |
| Claude | `claude-sonnet-5`, `claude-haiku-4-5-20251001` |

All runs used each provider's default reasoning setting and a 16,384 output-token limit.

When all 12 runs (6 models × 2 resolutions) are finished, print Tables 3 and 4:

```bash
python make_tables.py
```

## Scoring

The scoring code is in `evaluator.py`:

- **Parsing:** the model returns `{"texts": [...]}`. Every string is extracted, and repeated whitespace and
  line breaks are collapsed. If the model splits a text into single words, the words joined together are
  also added as a candidate.
- **Matching:** each output is assigned to at most one layer. A single output goes to the layer it is closer
  to. With several outputs, the contour/shading pair with the highest combined similarity is used.
- **EM:** the matched output equals the target text after lowercasing and whitespace normalization.
- **LS:** `1 − Levenshtein(target, output) / max(len(target), len(output))`. A layer with no matched output
  scores 0.
- **Avg. extracted texts (Table 4):** the number of parsed outputs per response.

## Results

Exact match (%) and Levenshtein similarity for each text layer (Table 3):

| Model | Prompt | 512 Contour EM | LS | 512 Shading EM | LS | 64 Contour EM | LS | 64 Shading EM | LS |
|---|---|---|---|---|---|---|---|---|---|
| Gemini 3.5 Flash-Lite | Naive | 93.0 | 0.981 | 0.0 | 0.010 | 0.0 | 0.000 | 100.0 | 1.000 |
| | Guided | 94.0 | 0.992 | 0.0 | 0.256 | 0.0 | 0.028 | 98.7 | 0.996 |
| Gemini 3.6 Flash | Naive | 96.3 | 0.992 | 0.0 | 0.001 | 0.0 | 0.000 | 100.0 | 1.000 |
| | Guided | 96.0 | 0.997 | 0.7 | 0.320 | 0.0 | 0.117 | 100.0 | 1.000 |
| GPT-5.6 Luna | Naive | 80.3 | 0.884 | 0.0 | 0.055 | 0.0 | 0.000 | 98.0 | 0.997 |
| | Guided | 99.7 | 1.000 | 1.0 | 0.269 | 0.0 | 0.129 | 97.3 | 0.994 |
| GPT-5.6 Terra | Naive | 97.7 | 0.986 | 0.0 | 0.009 | 0.0 | 0.001 | 95.0 | 0.990 |
| | Guided | 99.3 | 0.999 | 0.3 | 0.078 | 0.0 | 0.029 | 95.0 | 0.980 |
| Claude Haiku 4.5 | Naive | 96.3 | 0.995 | 0.0 | 0.002 | 0.0 | 0.010 | 93.7 | 0.975 |
| | Guided | 97.0 | 0.994 | 0.0 | 0.013 | 0.0 | 0.030 | 98.3 | 0.997 |
| Claude Sonnet 5 | Naive | 98.7 | 0.997 | 0.0 | 0.000 | 0.0 | 0.000 | 100.0 | 1.000 |
| | Guided | 96.7 | 0.990 | 0.0 | 0.006 | 0.0 | 0.003 | 98.7 | 0.997 |

Human readers (10 participants) reached 99.7% / 96.3% EM on contour / shading at 512×512 and 0.0% / 98.7% at
64×64. The human study data is not part of this repository.

## Repository layout

```
decoybench.json          dataset: image, contour text, shading text
dataset_stats.py         Table 1 statistics
resize_images.py         makes the 512x512 and 64x64 versions
evaluator.py             prompts, parsing and scoring
batch_generator.py       builds batch request files
run_batch_job.py         submits, monitors and downloads batch jobs
parse_batch_results.py   scores downloaded batch results
benchmark.py             direct (non-batch) evaluation
providers.py             API clients for the direct runs
pricing.py               token prices used for the cost report
make_tables.py           prints Tables 3 and 4
```

## Citation


The images were made with Decoy Font by Eric Lu (Mixfont):
https://www.mixfont.com/experiments/decoy-font
