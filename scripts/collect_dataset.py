#!/usr/bin/env python3
"""Collect a reproducible action-recognition dataset from Wikimedia Commons.

The script downloads 120 usable photographs for each class and saves the
corresponding source page, author, and license in data/manifest.csv.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import random
import re
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlencode

from PIL import Image, ImageOps


API_URL = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "ActionRecognitionCourseProject/1.0 (educational use)"
CLASS_CATEGORIES = {
    "sitting": ["People sitting"],
    "standing": ["People standing"],
    "waving": ["Female people waving hands", "Male people waving hands"],
}


def clean_html(value: str) -> str:
    value = re.sub(r"<[^>]+>", " ", value or "")
    return " ".join(html.unescape(value).split())


def curl_json(params: dict[str, str]) -> dict:
    command = [
        "curl",
        "-fsSL",
        "--retry",
        "8",
        "--retry-all-errors",
        "--connect-timeout",
        "20",
        "--max-time",
        "90",
        "-A",
        USER_AGENT,
        f"{API_URL}?{urlencode(params)}",
    ]
    result = subprocess.run(command, check=True, capture_output=True)
    return json.loads(result.stdout)


def category_candidates(category: str, maximum: int = 500) -> list[dict]:
    records: list[dict] = []
    continuation: dict[str, str] = {}
    while len(records) < maximum:
        params = {
            "action": "query",
            "format": "json",
            "formatversion": "2",
            "generator": "categorymembers",
            "gcmtitle": f"Category:{category}",
            "gcmnamespace": "6",
            "gcmtype": "file",
            "gcmlimit": "500",
            "prop": "imageinfo",
            "iiprop": "url|extmetadata|mime|size|sha1",
            "iiurlwidth": "640",
            **continuation,
        }
        try:
            payload = curl_json(params)
        except subprocess.CalledProcessError:
            if records:
                break
            raise
        for page in payload.get("query", {}).get("pages", []):
            info = (page.get("imageinfo") or [{}])[0]
            mime = info.get("mime", "")
            if mime in {"image/jpeg", "image/png", "image/webp"}:
                records.append({"title": page.get("title", ""), **info})
        if "continue" not in payload:
            break
        continuation = {
            key: str(value) for key, value in payload["continue"].items()
        }
    return records[:maximum]


def download(url: str, destination: Path) -> bool:
    result = subprocess.run(
        [
            "curl",
            "-fsSL",
            "--retry",
            "1",
            "--retry-all-errors",
            "--connect-timeout",
            "8",
            "--max-time",
            "15",
            "-A",
            USER_AGENT,
            "-o",
            str(destination),
            url,
        ],
        capture_output=True,
    )
    return result.returncode == 0


def difference_hash(image: Image.Image) -> int:
    gray = ImageOps.grayscale(image).resize((9, 8), Image.Resampling.LANCZOS)
    pixels = list(gray.getdata())
    bits = 0
    for row in range(8):
        for col in range(8):
            left = pixels[row * 9 + col]
            right = pixels[row * 9 + col + 1]
            bits = (bits << 1) | int(left > right)
    return bits


def metadata_value(metadata: dict, key: str) -> str:
    return clean_html(metadata.get(key, {}).get("value", ""))


def collect(output: Path, per_class: int, seed: int) -> None:
    randomizer = random.Random(seed)
    output.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, str | int]] = []
    global_sha256: set[str] = set()

    for class_name, categories in CLASS_CATEGORIES.items():
        class_dir = output / "raw" / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        candidates: list[dict] = []
        for category in categories:
            candidates.extend(category_candidates(category))
        randomizer.shuffle(candidates)

        accepted_hashes: list[int] = []
        accepted = 0
        attempts = 0
        for batch_start in range(0, len(candidates), 24):
            if accepted >= per_class:
                break
            batch = candidates[batch_start : batch_start + 24]
            with tempfile.TemporaryDirectory() as temp_dir_name:
                temp_dir = Path(temp_dir_name)

                def fetch(item):
                    index, candidate = item
                    image_url = candidate.get("thumburl") or candidate.get("url")
                    temp_path = temp_dir / f"{index:03d}.image"
                    success = bool(image_url) and download(image_url, temp_path)
                    return candidate, image_url, temp_path, success

                with ThreadPoolExecutor(max_workers=8) as executor:
                    downloaded = list(executor.map(fetch, enumerate(batch)))

                for candidate, image_url, temp_path, success in downloaded:
                    if accepted >= per_class:
                        break
                    attempts += 1
                    if not success:
                        continue
                    try:
                        raw_bytes = temp_path.read_bytes()
                        digest = hashlib.sha256(raw_bytes).hexdigest()
                        if digest in global_sha256:
                            continue
                        with Image.open(temp_path) as source:
                            image = ImageOps.exif_transpose(source).convert("RGB")
                            width, height = image.size
                            if min(width, height) < 180:
                                continue
                            perceptual = difference_hash(image)
                            if any((perceptual ^ old).bit_count() <= 3 for old in accepted_hashes):
                                continue
                            filename = f"{class_name}_{accepted + 1:03d}.jpg"
                            destination = class_dir / filename
                            image.thumbnail((640, 640), Image.Resampling.LANCZOS)
                            image.save(destination, "JPEG", quality=90, optimize=True)
                        metadata = candidate.get("extmetadata", {})
                        page_url = candidate.get("descriptionurl", "")
                        manifest_rows.append(
                            {
                                "filename": f"raw/{class_name}/{filename}",
                                "class": class_name,
                                "source_page": page_url,
                                "image_url": image_url,
                                "title": candidate.get("title", ""),
                                "author": metadata_value(metadata, "Artist") or metadata_value(metadata, "Credit"),
                                "license": metadata_value(metadata, "LicenseShortName"),
                                "license_url": metadata_value(metadata, "LicenseUrl"),
                                "original_width": width,
                                "original_height": height,
                                "sha256": digest,
                            }
                        )
                        accepted_hashes.append(perceptual)
                        global_sha256.add(digest)
                        accepted += 1
                        print(f"{class_name}: {accepted}/{per_class}", flush=True)
                    except (OSError, ValueError):
                        continue

        if accepted < per_class:
            raise RuntimeError(
                f"Only {accepted} valid images found for {class_name} after {attempts} attempts"
            )

    manifest_path = output / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)

    summary = {
        "source": "Wikimedia Commons",
        "collection_method": "MediaWiki API category members",
        "seed": seed,
        "classes": {
            name: sum(row["class"] == name for row in manifest_rows)
            for name in CLASS_CATEGORIES
        },
        "total_images": len(manifest_rows),
    }
    (output / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data"))
    parser.add_argument("--per-class", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    collect(args.output, args.per_class, args.seed)


if __name__ == "__main__":
    main()
