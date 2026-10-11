"""Step 0: prepare paired U/G counting inputs without loading Show-o2."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import re
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
WORDS = {1: "one", 2: "two", 3: "three", 4: "four"}
CLASSES = {"dog": "dogs", "cat": "cats", "bear": "bears",
           "elephant": "elephants", "cow": "cows", "zebra": "zebras",
           "giraffe": "giraffes", "chair": "chairs", "bottle": "bottles",
           "orange": "oranges"}
GENEVAL_URL = "https://raw.githubusercontent.com/djghosh13/geneval/main/prompts/evaluation_metadata.jsonl"
COCO_HOST = "https://s3.amazonaws.com/images.cocodataset.org"


def normalized(text):
    return " ".join(text.split()).casefold().rstrip(".")


def read_records(path):
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    data = json.loads(text)
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a list of input records")
    return data


def exclusions(paths):
    image_ids, prompts = set(), set()
    for path in paths:
        for entry in read_records(path):
            record = entry.get("source_record", entry)
            for field in ("source_url", "image_path"):
                match = re.search(r"(?<!\d)(\d{12})\.(?:jpg|png)", record.get(field, ""))
                if match:
                    image_ids.add(int(match[1]))
            if "prompt" in record and (entry.get("task") == "G" or "image_path" not in record):
                prompts.add(normalized(record["prompt"]))
    return image_ids, prompts


def select_understanding(coco, excluded_ids, per_count=10, seed=20261010):
    images = {row["id"]: row for row in coco["images"]}
    names = {row["id"]: row["name"] for row in coco["categories"]}
    grouped = defaultdict(list)
    for annotation in coco["annotations"]:
        if names[annotation["category_id"]] in CLASSES:
            grouped[annotation["image_id"], annotation["category_id"]].append(annotation)
    candidates = defaultdict(list)
    for (image_id, category_id), annotations in sorted(grouped.items()):
        image = images[image_id]
        count = len(annotations)
        if image_id in excluded_ids or count not in WORDS:
            continue
        # Avoid crowd labels and very small instances; review is still required.
        if any(a.get("iscrowd", 0) or a["area"] < .01 * image["width"] * image["height"]
               for a in annotations):
            continue
        name = names[category_id]
        candidates[count].append(dict(
            coco_image_id=image_id, object_class=name, target_count=count,
            prompt=f"How many {CLASSES[name]} are there?", expected_answer=WORDS[count],
            image_path=f"images/{image_id:012d}.jpg",
            source_url=f"{COCO_HOST}/val2017/{image_id:012d}.jpg",
            annotation_ids=sorted(a["id"] for a in annotations),
            reference_source="COCO instances_val2017", review_status="pending"))
    rng, selected, used, class_counts = random.Random(seed), [], set(excluded_ids), Counter()
    for count in WORDS:
        pool = candidates[count]
        rng.shuffle(pool)
        for _ in range(per_count):
            available = [row for row in pool if row["coco_image_id"] not in used]
            if not available:
                raise ValueError(f"Not enough unique U images with count {count}")
            row = min(available, key=lambda x: class_counts[x["object_class"]])
            row["id"] = f"U_{len(selected):04d}"
            selected.append(row)
            used.add(row["coco_image_id"])
            class_counts[row["object_class"]] += 1
    return selected


def select_generation(metadata, excluded_prompts, n=10, seed=20261010):
    pools = defaultdict(list)
    seen = set(excluded_prompts)
    for index, record in enumerate(metadata):
        prompt = normalized(record["prompt"])
        if record["tag"] == "counting" and prompt not in seen:
            seen.add(prompt)
            pools[record["include"][0]["count"]].append((index, record))
    rng = random.Random(seed)
    for pool in pools.values():
        rng.shuffle(pool)
    selected, counts = [], sorted(pools)
    if not counts:
        raise ValueError("No unused GenEval counting prompts")
    for i in range(n):
        count = counts[i % len(counts)]
        if not pools[count]:
            raise ValueError(f"Not enough unused GenEval prompts with count {count}")
        index, record = pools[count].pop()
        selected.append(dict(id=f"G_{i:04d}", geneval_index=index, metadata=record))
    return selected


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def prepare(coco_path, metadata_path, output, used_paths=(), per_count=10, n_prompts=10):
    # Exclude all known pilot inputs, including the already inspected test split.
    known = sorted((ROOT / "prompts").glob("understanding_*.json"))
    known.append(ROOT / "prompts/generation_counting_spatial_40.json")
    paths = known + list(used_paths)
    excluded_ids, excluded_prompts = exclusions(paths)
    u = select_understanding(json.loads(coco_path.read_text(encoding="utf-8")),
                             excluded_ids, per_count)
    g = select_generation(read_records(metadata_path), excluded_prompts, n_prompts)
    seeds = [1001, 1002, 1003, 1004]
    cases = [dict(id=f"{row['id']}_s{seed}", prompt_id=row["id"], seed=seed,
                  prompt=row["metadata"]["prompt"],
                  relative_output=f"{i:05d}/samples/{j:04d}.png")
             for i, row in enumerate(g) for j, seed in enumerate(seeds)]
    # One randomly chosen seed per prompt: ten paired cases in the default pilot.
    rng = random.Random(20261011)
    audit = [rng.choice([case for case in cases if case["prompt_id"] == row["id"]])["id"]
             for row in rng.sample(g, min(10, len(g)))]
    output.mkdir(parents=True, exist_ok=True)
    if (output / "protocol.json").exists():
        raise FileExistsError("Test already prepared; use a new output directory to change it.")
    write_json(output / "understanding.json", u)
    write_json(output / "generation.json", g)
    write_json(output / "generation_cases.json", cases)
    write_json(output / "human_audit_cases.json", audit)
    (output / "evaluation_metadata.jsonl").write_text(
        "".join(json.dumps(row["metadata"]) + "\n" for row in g), encoding="utf-8")
    write_json(output / "protocol.json", dict(
        schema_version=1, selection_seed=20261010, generation_seeds=seeds,
        u_images=len(u), g_images_per_mode=len(cases),
        modes=["dense", "shared", "specialized", "depth"],
        metrics={"U": "exact integer count accuracy", "G": "GenEval counting correct rate",
                 "visual_quality": "blind paired A/B/tie audit, separate from counting"},
        state="pending_reference_review_and_training_provenance_check",
        exclusion_files=[str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
                         for path in paths],
        excluded_coco_images=len(excluded_ids), excluded_generation_prompts=len(excluded_prompts),
        sources={"G": GENEVAL_URL, "U": "COCO instances_val2017.json"},
        source_sha256={"G": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
                       "U": hashlib.sha256(coco_path.read_bytes()).hexdigest()},
        timing="Later: warm each mode; CUDA sync; time inference including routing and VAE, "
               "excluding load/download, file saving and evaluation. U includes answer decode.",
        decisions="Freeze weights, routing settings, precision, resolution, guidance and "
                  "sampling before running; use identical prompt/image/seed across modes."))
    return u, g


def download_images(records, output):
    folder = output / "images"
    folder.mkdir(exist_ok=True)
    for row in records:
        path = output / row["image_path"]
        if path.exists():
            continue
        partial = path.with_suffix(".part")
        urllib.request.urlretrieve(row["source_url"], partial)
        partial.replace(path)
        print(f"Downloaded {row['id']}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coco-annotations", type=Path, required=True)
    parser.add_argument("--geneval-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--used-records", type=Path, nargs="*", default=[])
    parser.add_argument("--download-images", action="store_true")
    args = parser.parse_args()
    u, g = prepare(args.coco_annotations, args.geneval_metadata, args.output, args.used_records)
    if args.download_images:
        download_images(u, args.output)
    print(f"Prepared {len(u)} U images; {len(g)} G prompts x 4 seeds. References need review.")
