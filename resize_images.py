import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image

EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}


def resize(job):
    src, dst, size = job
    with Image.open(src) as img:
        img.resize((size, size), Image.Resampling.LANCZOS).save(dst)


def main():
    parser = argparse.ArgumentParser(description="Downscale the original images to a square resolution.")
    parser.add_argument("--src-dir", default="images/original")
    parser.add_argument("--dst-dir", default="images/512x512")
    parser.add_argument("--size", type=int, default=512)
    args = parser.parse_args()

    src_dir, dst_dir = Path(args.src_dir), Path(args.dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(p, dst_dir / p.name, args.size) for p in src_dir.iterdir() if p.suffix.lower() in EXTENSIONS]

    with ProcessPoolExecutor() as pool:
        list(pool.map(resize, jobs))
    print(f"resized {len(jobs)} images to {args.size}x{args.size} in {dst_dir}")


if __name__ == "__main__":
    main()
