import argparse
import json
import os
import time

from dotenv import load_dotenv

from evaluator import get_model_dir
from providers import detect_provider

load_dotenv()


class OpenAIBatches:
    def __init__(self, model):
        from openai import OpenAI
        self.client = OpenAI()

    def submit(self, path):
        with open(path, "rb") as f:
            file = self.client.files.create(file=f, purpose="batch")
        batch = self.client.batches.create(
            input_file_id=file.id, endpoint="/v1/chat/completions", completion_window="24h")
        return batch.id

    def status(self, batch_id):
        batch = self.client.batches.retrieve(batch_id)
        counts = batch.request_counts
        text = f"{batch.status} {counts.completed}/{counts.total}" if counts else batch.status
        if counts and counts.failed:
            text += f" ({counts.failed} failed)"
        return text, batch.status in ("completed", "failed", "cancelled", "expired")

    def download(self, batch_id, path):
        batch = self.client.batches.retrieve(batch_id)
        if not batch.output_file_id:
            print(f"{batch_id}: no output file (error file: {batch.error_file_id})")
            return False
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.client.files.content(batch.output_file_id).text)
        return True


class AnthropicBatches:
    def __init__(self, model):
        import anthropic
        self.client = anthropic.Anthropic(timeout=600.0)

    def submit(self, path):
        with open(path, encoding="utf-8") as f:
            requests = [json.loads(line) for line in f if line.strip()]
        return self.client.messages.batches.create(requests=requests).id

    def status(self, batch_id):
        batch = self.client.messages.batches.retrieve(batch_id)
        c = batch.request_counts
        text = f"{batch.processing_status} ok={c.succeeded} running={c.processing} errored={c.errored}"
        return text, batch.processing_status == "ended"

    def download(self, batch_id, path):
        with open(path, "w", encoding="utf-8") as f:
            for result in self.client.messages.batches.results(batch_id):
                f.write(result.model_dump_json() + "\n")
        return True


class GoogleBatches:
    def __init__(self, model):
        from google import genai
        from google.genai import types
        self.types = types
        self.client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
        self.model = model if model.startswith("models/") else f"models/{model}"

    def submit(self, path):
        config = self.types.UploadFileConfig(mime_type="application/jsonl")
        uploaded = self.client.files.upload(file=path, config=config)
        job = self.client.batches.create(model=self.model, src=uploaded.name,
                                         config={"display_name": os.path.basename(path)})
        return job.name

    def status(self, batch_id):
        state = self.client.batches.get(name=batch_id).state.name
        return state, any(s in state for s in ("SUCCEEDED", "FAILED", "CANCELLED", "EXPIRED"))

    def download(self, batch_id, path):
        job = self.client.batches.get(name=batch_id)
        if not job.dest or not job.dest.file_name:
            print(f"{batch_id}: no result file ({job.state.name})")
            return False
        content = self.client.files.download(file=job.dest.file_name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content.decode("utf-8") if isinstance(content, bytes) else content)
        return True


BACKENDS = {"openai": OpenAIBatches, "anthropic": AnthropicBatches, "google": GoogleBatches}


def split_jsonl(path, parts):
    with open(path, encoding="utf-8") as f:
        lines = [line if line.endswith("\n") else line + "\n" for line in f if line.strip()]
    if parts <= 1:
        return [path]
    size = -(-len(lines) // parts)
    base, ext = os.path.splitext(path)
    chunks = []
    for i in range(0, len(lines), size):
        chunk = f"{base}_part{len(chunks) + 1}{ext}"
        with open(chunk, "w", encoding="utf-8") as f:
            f.writelines(lines[i:i + size])
        chunks.append(chunk)
    return chunks


def wait_for(backend, batch_ids, poll_interval):
    start = time.time()
    pending = set(batch_ids)
    while pending:
        lines = []
        for i, batch_id in enumerate(batch_ids, 1):
            if batch_id not in pending:
                lines.append(f"#{i} done")
                continue
            try:
                text, finished = backend.status(batch_id)
            except Exception as e:
                text, finished = f"error: {e}", False
            lines.append(f"#{i} {text}")
            if finished:
                pending.discard(batch_id)
        elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start))
        print(f"[{elapsed}] " + " | ".join(lines))
        if pending:
            time.sleep(poll_interval)


def download_all(backend, batch_ids, output):
    if len(batch_ids) == 1:
        backend.download(batch_ids[0], output)
        return
    base, ext = os.path.splitext(output)
    parts = []
    for i, batch_id in enumerate(batch_ids, 1):
        part = f"{base}_part{i}{ext}"
        if backend.download(batch_id, part):
            parts.append(part)
    with open(output, "w", encoding="utf-8") as out:
        for part in parts:
            with open(part, encoding="utf-8") as f:
                out.writelines(line if line.endswith("\n") else line + "\n" for line in f if line.strip())
    print(f"merged {len(parts)}/{len(batch_ids)} parts into {output}")


def main():
    parser = argparse.ArgumentParser(description="Submit, monitor and download batch jobs.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--provider", choices=list(BACKENDS))
    parser.add_argument("--output-dir")
    parser.add_argument("--input-jsonl")
    parser.add_argument("--output-jsonl")
    parser.add_argument("--num-chunks", type=int, default=6)
    parser.add_argument("--batch-id", help="existing batch id(s), comma separated")
    parser.add_argument("--poll-interval", type=int, default=30)
    parser.add_argument("--action", choices=["auto", "submit", "status", "download"], default="auto")
    args = parser.parse_args()

    out_dir = args.output_dir or get_model_dir(args.model)
    input_jsonl = args.input_jsonl or os.path.join(out_dir, "batch_requests.jsonl")
    output_jsonl = args.output_jsonl or os.path.join(out_dir, "batch_results.jsonl")
    provider = args.provider or detect_provider(args.model)
    backend = BACKENDS[provider](args.model)

    if args.batch_id:
        batch_ids = [b.strip() for b in args.batch_id.split(",") if b.strip()]
    elif args.action in ("status", "download"):
        parser.error(f"--batch-id is required for --action {args.action}")
    else:
        batch_ids = []
        for chunk in split_jsonl(input_jsonl, args.num_chunks):
            batch_ids.append(backend.submit(chunk))
            print(f"submitted {chunk} -> {batch_ids[-1]}")
        print(f"batch ids: {','.join(batch_ids)}")

    if args.action == "submit":
        return
    if args.action == "status":
        for batch_id in batch_ids:
            print(batch_id, backend.status(batch_id)[0])
        return
    if args.action == "auto":
        wait_for(backend, batch_ids, args.poll_interval)
    download_all(backend, batch_ids, output_jsonl)


if __name__ == "__main__":
    main()
