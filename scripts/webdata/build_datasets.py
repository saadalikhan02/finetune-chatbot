#!/usr/bin/env python3
"""Merge all generated QA banks into final train/validation/test JSONL
datasets in the Gemma fine-tuning project's message format.

Splitting strategy (see README section "Dataset splitting" for the full
rationale):

  - Adversarial/hallucination-trap examples (data/knowledge/qa_adversarial.jsonl)
    are test-only by construction - they never appear in train or validation.
  - Every other record is grouped into a cluster with every other record
    that cites at least one of the same fact_id(s) (via union-find), so
    near-duplicate phrasings of the same underlying fact/answer always land
    in the same split - this is what avoids data leakage. Records that cite
    no facts (unknown-information, out-of-scope, conversational, ambiguous)
    are each their own cluster, since there's no cross-question overlap risk
    there.
  - Clusters are shuffled with a fixed seed and greedily assigned to
    train/validation/test-pool to hit the configured ratios by record count.
  - Finally, the script guarantees every QA category present in the pool
    has at least one representative in the test set (moving one cluster
    from train if a category would otherwise be test-set-absent) - so the
    test set deliberately covers every category rather than being an
    arbitrary random sample (see spec: "the test set must be special").

Usage:
    python scripts/webdata/build_datasets.py
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from website_pipeline.models import read_jsonl, write_jsonl  # noqa: E402

SYSTEM_PROMPT = (
    "You are the official AI assistant for Technyx Systems. Answer questions about Technyx "
    "Systems using verified information available from the company's official website. Do not "
    "invent or assume information. If the requested information is not available, clearly say "
    "that you do not have verified information about it."
)

DEFAULT_QA_FILES = [
    "data/knowledge/qa_auto.jsonl",
    "data/knowledge/qa_curated.jsonl",
    "data/knowledge/qa_followups.jsonl",
    "data/knowledge/qa_unknown.jsonl",
    "data/knowledge/qa_out_of_scope.jsonl",
    "data/knowledge/qa_conversational.jsonl",
]
ADVERSARIAL_FILE = "data/knowledge/qa_adversarial.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa-files", nargs="*", default=DEFAULT_QA_FILES)
    parser.add_argument("--adversarial-file", default=ADVERSARIAL_FILE)
    parser.add_argument("--output-dir", default="data/datasets", type=Path)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42, help="Deterministic split seed - documented here")
    return parser.parse_args()


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: str, y: str) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx != ry:
            self.parent[rx] = ry


def build_clusters(records: list[dict[str, Any]]) -> dict[str, list[int]]:
    uf = UnionFind()
    for r in records:
        fact_ids = r.get("fact_ids") or []
        for fid in fact_ids[1:]:
            uf.union(fact_ids[0], fid)

    clusters: dict[str, list[int]] = {}
    for i, r in enumerate(records):
        fact_ids = r.get("fact_ids") or []
        if fact_ids:
            key = "fact:" + uf.find(fact_ids[0])
        else:
            key = f"solo:{i}"
        clusters.setdefault(key, []).append(i)
    return clusters


def to_gemma_record(qa: dict[str, Any]) -> dict[str, Any]:
    messages = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}]
    for turn in qa.get("history") or []:
        messages.append(turn)
    messages.append({"role": "user", "content": [{"type": "text", "text": qa["question"]}]})
    messages.append({"role": "assistant", "content": [{"type": "text", "text": qa["answer"]}]})

    metadata = {
        "source_url": qa.get("source_url"),
        "source_page_title": qa.get("source_page_title"),
        "source_section": qa.get("source_section"),
        "source_excerpt": qa.get("source_excerpt"),
        "grounding_status": qa["grounding_status"],
        "category": qa["category"],
        "fact_ids": qa.get("fact_ids") or [],
    }
    if qa.get("expected_behavior"):
        metadata["expected_behavior"] = qa["expected_behavior"]

    return {"messages": messages, "metadata": metadata}


def main() -> int:
    args = parse_args()

    records: list[dict[str, Any]] = []
    for path in args.qa_files:
        records.extend(read_jsonl(path))

    adversarial_records = read_jsonl(args.adversarial_file) if Path(args.adversarial_file).exists() else []

    clusters = build_clusters(records)
    cluster_keys = list(clusters.keys())
    rng = random.Random(args.seed)
    rng.shuffle(cluster_keys)

    total = len(records)
    train_target = round(total * args.train_ratio)
    val_target = round(total * args.validation_ratio)

    train_idx: list[int] = []
    val_idx: list[int] = []
    test_pool_idx: list[int] = []

    for key in cluster_keys:
        members = clusters[key]
        if len(train_idx) < train_target:
            train_idx.extend(members)
        elif len(val_idx) < val_target:
            val_idx.extend(members)
        else:
            test_pool_idx.extend(members)

    def categories_in(indices: list[int]) -> set[str]:
        return {records[i]["category"] for i in indices}

    all_categories = categories_in(range(total))
    test_categories = categories_in(test_pool_idx)
    missing = all_categories - test_categories

    for category in sorted(missing):
        # Find a cluster (currently in train, else validation) containing
        # this category and move its whole cluster into the test pool so
        # every category is represented in the "special" test set.
        moved = False
        for pool_name, pool in (("train", train_idx), ("validation", val_idx)):
            for key in cluster_keys:
                members = clusters[key]
                if all(m in pool for m in members) and any(
                    records[m]["category"] == category for m in members
                ):
                    for m in members:
                        pool.remove(m)
                    test_pool_idx.extend(members)
                    moved = True
                    break
            if moved:
                break

    train_records = [records[i] for i in train_idx]
    val_records = [records[i] for i in val_idx]
    test_records = [records[i] for i in test_pool_idx] + adversarial_records

    gemma_train = [to_gemma_record(r) for r in train_records]
    gemma_val = [to_gemma_record(r) for r in val_records]
    gemma_test = [to_gemma_record(r) for r in test_records]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "train.jsonl", gemma_train)
    write_jsonl(args.output_dir / "validation.jsonl", gemma_val)
    write_jsonl(args.output_dir / "test.jsonl", gemma_test)

    print(f"Split seed: {args.seed} (deterministic)")
    print(f"Total pre-split QA records: {total} (+ {len(adversarial_records)} adversarial, test-only)")
    print(f"Train:      {len(gemma_train)}")
    print(f"Validation: {len(gemma_val)}")
    print(f"Test:       {len(gemma_test)} ({len(test_pool_idx)} pooled + {len(adversarial_records)} adversarial)")

    print("\nCategory coverage in test set:")
    test_cats = categories_in(test_pool_idx) | {r["category"] for r in adversarial_records}
    for cat in sorted(all_categories | {r["category"] for r in adversarial_records}):
        marker = "yes" if cat in test_cats else "MISSING"
        print(f"  {cat:20s} {marker}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
