"""Download a balanced VulnLLM-R training sample and prepare Joern AST labels."""

import argparse
import hashlib
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from train_lstm import export_with_docker, positive_int

from salomon.adapters.lstm.checkpoint import REPRESENTATION

DATASET = "UCSB-SURFI/VulnLLM-R-Train-Data"


def fetch_json(url: str) -> dict:
    for attempt in range(4):
        try:
            request = Request(url, headers={"User-Agent": "salomon-lstm-data/1.0"})
            with urlopen(request, timeout=90) as response:
                return json.load(response)
        except (HTTPError, URLError, TimeoutError) as error:
            if isinstance(error, HTTPError) and error.code not in (429, 500, 502, 503, 504):
                raise
            if attempt == 3:
                raise
            time.sleep(2**attempt)
    raise RuntimeError("Download failed")


def save_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def download_sample(output: Path, count: int, seed: int) -> list[dict]:
    cache = output / "sample.json"
    if cache.exists():
        saved = json.loads(cache.read_text(encoding="utf-8"))
        if saved["requested_samples"] != count or saved["seed"] != seed:
            raise ValueError("Cached sample has different settings; use another --output directory.")
        return saved["rows"]
    metadata = fetch_json(f"https://huggingface.co/api/datasets/{DATASET}")

    def page(offset: int) -> dict:
        query = urlencode(
            {"dataset": DATASET, "config": "default", "split": "function_level", "offset": offset, "length": 100}
        )
        return fetch_json("https://datasets-server.huggingface.co/rows?" + query)

    first = page(0)
    offsets = list(range(0, first["num_rows_total"], 100))
    random.Random(seed).shuffle(offsets)
    rows, seen = [], set()
    counts = Counter()
    for offset in offsets:
        batch = first if offset == 0 else page(offset)
        for item in batch["rows"]:
            row = item["row"]
            code, label = row.get("code"), row.get("target")
            if (
                item.get("truncated_cells")
                or row.get("language", "").lower() not in {"c", "c++", "cpp"}
                or row.get("if_multi_func")
                or row.get("stack_trace")
                or not isinstance(code, str)
                or not code.strip()
                or type(label) is not int
                or label not in (0, 1)
            ):
                continue
            digest = hashlib.sha256(code.encode()).hexdigest()
            identity = (digest, label)
            if identity in seen or counts[label] >= count // 2:
                continue
            seen.add(identity)
            rows.append({**row, "row_index": item["row_idx"], "code_sha256": digest})
            counts[label] += 1
        print(f"Downloaded: safe={counts[0]} vulnerable={counts[1]}", flush=True)
        if len(rows) == count:
            break
    if len(rows) != count:
        raise ValueError(f"Only found {len(rows)} suitable rows; requested {count}.")
    save_json(
        cache,
        {
            "dataset": DATASET,
            "revision": metadata["sha"],
            "split": "function_level",
            "requested_samples": count,
            "seed": seed,
            "rows": rows,
        },
    )
    return rows


def prepare_labels(rows: list[dict], records: list[dict], seed: int, max_length: int = 512) -> tuple[list[dict], dict]:
    by_file = defaultdict(list)
    for record in records:
        by_file[Path(record["source"].replace("\\", "/")).name].append(record)
    selected, skipped = [], []
    for row in rows:
        source = f"sample_{row['row_index']}.cpp"
        methods = by_file[source]
        if len(methods) != 1 or not methods[0]["tokens"]:
            skipped.append({"source": source, "exported_functions": len(methods)})
            continue
        selected.append((row, methods[0], source))

    # Conservatively join matching function names and identical ASTs, transitively.
    # The dataset often omits project provenance; this is not a project-held-out split.
    parents = list(range(len(selected)))

    def root(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    keys = {}
    for index, (_, record, _) in enumerate(selected):
        ast_hash = hashlib.sha256(json.dumps(record["tokens"][:max_length]).encode()).hexdigest()
        for key in (("name", record["function"]), ("ast", ast_hash)):
            if key in keys:
                parents[root(index)] = root(keys[key])
            keys[key] = index
    groups = defaultdict(list)
    for index in range(len(selected)):
        groups[root(index)].append(index)
    shuffled = list(groups.values())
    random.Random(seed).shuffle(shuffled)
    totals = Counter(row["target"] for row, _, _ in selected)
    validation = Counter()
    labels = []
    for members in shuffled:
        counts = Counter(selected[index][0]["target"] for index in members)
        before = sum((validation[label] - totals[label] * 0.2) ** 2 for label in (0, 1))
        after = sum((validation[label] + counts[label] - totals[label] * 0.2) ** 2 for label in (0, 1))
        split = "validation" if after < before else "train"
        if split == "validation":
            validation.update(counts)
        for index in members:
            row, record, source = selected[index]
            labels.append(
                {
                    "source": source,
                    "function": record["function"],
                    "label": row["target"],
                    "split": split,
                    "group": f"related-{min(members)}",
                }
            )
    split_counts = {
        split: dict(Counter(entry["label"] for entry in labels if entry["split"] == split))
        for split in ("train", "validation")
    }
    if any(set(counts) != {0, 1} for counts in split_counts.values()):
        raise ValueError("Could not build train and validation splits containing both classes.")
    return labels, {
        "downloaded": len(rows),
        "usable": len(labels),
        "skipped": skipped,
        "split_class_counts": split_counts,
        "groups": len(groups),
        "max_length": max_length,
        "grouping": "function name and identical model input, transitively; project provenance unavailable",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/processed/vulnllm-lstm-vars-v2"))
    parser.add_argument("--samples", type=positive_int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=positive_int, default=512)
    parser.add_argument("--image", default="salomon-lstm:latest")
    parser.add_argument("--export-timeout", type=positive_int, default=1800)
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args()
    if args.samples < 4 or args.samples % 2:
        parser.error("--samples must be an even number >= 4")
    args.output.mkdir(parents=True, exist_ok=True)
    rows = download_sample(args.output, args.samples, args.seed)
    sources = args.output / "sources"
    sources.mkdir(exist_ok=True)
    for row in rows:
        (sources / f"sample_{row['row_index']}.cpp").write_text(row["code"], encoding="utf-8")
    if args.download_only:
        return
    sequence_path = args.output / "sequences.json"
    if sequence_path.exists():
        records = json.loads(sequence_path.read_text(encoding="utf-8"))
        if any(record.get("representation") != REPRESENTATION for record in records):
            raise ValueError("Cached AST representation is outdated; use a new --output directory and export again.")
    else:
        print("Exporting Joern AST; this can take several minutes...", flush=True)
        records = export_with_docker(sources, args.image, args.export_timeout)
        save_json(sequence_path, records)
    labels, report = prepare_labels(rows, records, args.seed, args.max_length)
    save_json(args.output / "labels.json", labels)
    save_json(args.output / "preparation.json", report)
    summary = {key: value for key, value in report.items() if key != "skipped"}
    summary["skipped_count"] = len(report["skipped"])
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
