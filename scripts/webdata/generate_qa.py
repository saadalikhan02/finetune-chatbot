#!/usr/bin/env python3
"""Auto-generate QA pairs from facts whose source is *already* a
question-and-answer pair on the website.

Technyx's site publishes real FAQ content in two places: the /faq page and
an "Everything You Need to Know" FAQ block on every /platforms/<name> page
(also present as schema.org FAQPage JSON-LD). Where a fact's
source_section is itself phrased as a question (e.g. "Does Technyx provide
ongoing Sitecore support and development after launch?"), the fact text is
already the site's own answer - turning it into a QA pair requires no
paraphrasing and carries essentially no grounding risk.

This script does NOT invent additional questions/phrasings - see
data/knowledge/qa_curated.jsonl for the hand-authored examples (paraphrases,
multi-fact questions, follow-ups, unknown-info, out-of-scope, adversarial)
that require actual judgment about phrasing and grounding.

Usage:
    python scripts/webdata/generate_qa.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from website_pipeline.models import QAExample, read_jsonl, write_jsonl  # noqa: E402

# Marketing headings like "Ready to Build What's Next?" or "Have Something
# Difficult to Deliver?" are CTAs that happen to end in "?" - they are not
# genuine FAQ questions and the "content" after them is a generic pitch, not
# an answer to that literal question. Requiring the heading to *start* with
# an interrogative word filters those out while keeping every real FAQ
# question (all of which are phrased this way: "What...", "Does...", etc).
INTERROGATIVE_STARTS = (
    "what", "why", "how", "does", "do", "can", "is", "are", "who", "where",
    "which", "should", "will", "did",
)


def _looks_like_real_question(section_heading: str) -> bool:
    first_word = section_heading.strip().split(" ", 1)[0].lower().strip("'’")
    return first_word in INTERROGATIVE_STARTS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--facts", default="data/knowledge/facts.jsonl", type=Path)
    parser.add_argument("--output", default="data/knowledge/qa_auto.jsonl", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    facts = read_jsonl(args.facts)

    examples: list[QAExample] = []
    for fact in facts:
        section = fact["source_section"].strip()
        if not section.endswith("?") or not _looks_like_real_question(section):
            continue

        category = "site_faq" if fact["source_url"].rstrip("/").endswith("/faq") else "platform_faq"
        examples.append(
            QAExample(
                question=section,
                answer=fact["fact"],
                source_url=fact["source_url"],
                source_page_title=fact["source_page_title"],
                source_section=fact["source_section"],
                source_excerpt=fact["source_excerpt"],
                grounding_status="grounded",
                category=category,
                fact_ids=[fact["fact_id"]],
            )
        )

    write_jsonl(args.output, examples)
    print(f"Generated {len(examples)} QA pair(s) from FAQ-shaped facts -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
