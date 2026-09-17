# Technyx Systems website dataset pipeline

This directory (plus `data/knowledge/` and `data/datasets/`) holds the
output of a crawl-to-dataset pipeline that turns the public
[technyxsystems.com](https://technyxsystems.com/) website into grounded
JSONL training data for the Gemma 3 1B fine-tuning project.

**The website is the only source of truth.** Every fact and every generated
question/answer pair traces back to a specific URL, section, and excerpt.
Nothing about Technyx (services, locations, clients, technologies, etc.) is
invented, inferred, or assumed beyond what these pages explicitly state.

## Pipeline stages

```
crawl_site.py        -> data/website/raw/<crawl_id>/pages.jsonl   (raw HTML snapshots)
clean_pages.py        -> data/website/clean/<document_id>.json     (structured sections)
extract_facts.py      -> data/knowledge/facts.jsonl                (atomic, classified, traceable facts)
generate_qa.py         -> data/knowledge/qa_auto.jsonl              (auto: site's own FAQ Q&A pairs)
seed_curated_qa.py      -> data/knowledge/qa_curated.jsonl           (paraphrases, multi-fact, company/service/industry QA)
seed_edge_case_qa.py     -> data/knowledge/qa_followups.jsonl         (multi-turn conversations)
                        -> data/knowledge/qa_unknown.jsonl           (verified-absent-info refusals)
                        -> data/knowledge/qa_out_of_scope.jsonl      (unrelated-question refusals)
                        -> data/knowledge/qa_conversational.jsonl    (greetings/thanks)
                        -> data/knowledge/qa_adversarial.jsonl       (test-only hallucination traps)
validate_grounding.py  -> (lint check across all qa_*.jsonl above)
build_datasets.py      -> data/datasets/{train,validation,test}.jsonl  (final Gemma-format datasets)
export_eval_cases.py   -> evaluation/test_cases.jsonl               (bridges to scripts/evaluate.py)
```

Re-running the whole pipeline from scratch:

```bash
python scripts/webdata/crawl_site.py
python scripts/webdata/clean_pages.py
python scripts/webdata/extract_facts.py
python scripts/webdata/generate_qa.py
python scripts/webdata/seed_curated_qa.py
python scripts/webdata/seed_edge_case_qa.py
python scripts/webdata/validate_grounding.py
python scripts/webdata/build_datasets.py
python scripts/webdata/export_eval_cases.py
```

`seed_curated_qa.py` and `seed_edge_case_qa.py` are **hand-authored**, not
template-generated - see "QA generation" below for why. If the site
changes, re-run the crawl/clean/fact steps, then review whether these two
scripts still cite accurate facts (`validate_grounding.py` will catch a
citation that no longer resolves to a real fact_id, but it can't tell you
whether a *new* fact on the site should be added as a new QA pair - that
still needs a human read-through).

## 1. Crawling (`crawl_site.py`, `src/website_pipeline/crawler.py`)

- Discovers pages from `/sitemap.xml` (33 URLs as of the last crawl - no
  sitemap index nesting was needed) plus BFS link-following from the
  homepage as a safety net, in case a page is missing from the sitemap.
- Respects `robots.txt` (`Allow: /` - no restrictions found).
- Normalizes URLs (`src/website_pipeline/url_utils.py`): forces https,
  drops `www.`, strips trailing slashes and tracking query parameters
  (`utm_*`, `fbclid`, etc.), so `https://technyxsystems.com/contact` and
  `https://www.technyxsystems.com/contact/?utm_source=x` dedupe correctly.
- Technyx's site has **no `/en/`-style locale prefix** - every page is a
  flat path (`/services`, `/industries/automotive`, etc.), all in English
  (confirmed per-page via `langdetect` in the clean stage). `PRIMARY_LANGUAGE`
  is configurable in `configs/website.yaml` if that ever changes.
- Plain HTTP (`httpx`) + BeautifulSoup only - no Playwright/browser
  rendering. Technyx's pages are server-rendered (verified: the real page
  text is present in the raw HTML response), so browser rendering isn't
  needed for this site.
- Each crawl gets its own timestamped `crawl_id` subdirectory under
  `data/website/raw/` - previous crawls are never overwritten.

## 2. Cleaning (`clean_pages.py`, `src/website_pipeline/html_clean.py`)

Turns raw HTML into `{url, title, sections: [{heading, content: [...]}]}`.
A few site-specific extraction traps this handles explicitly (see the
module docstring for detail):

- **Animated stat counters** ("30+ Clients", "10+ Years of Delivery", ...):
  the visible DOM contains every digit 0-9 for a CSS "odometer" animation;
  the real number only exists in a `data-rest` attribute. Decoded
  mechanically (`data-rest % 10` per digit), not guessed.
- **Cloudflare-obfuscated email**: `info@technyxsystems.com` is recovered
  by decoding the standard Cloudflare email-protection XOR cipher from the
  page's own `data-cfemail` attribute - not invented.
- **FAQ accordions**: answer text lives in plain `<div>`s, not `<p>`/`<li>`,
  so `<div>` is treated as a content tag (only when it's a text-bearing
  leaf, so layout-wrapper `<div>`s aren't captured as one giant blob).
- Duplicate nav/footer menus and duplicated marquee/carousel content are
  stripped/deduplicated so they never become "facts".
- schema.org JSON-LD (`Organization`, `WebPage`) is parsed for the
  official org name and social links.

## 3. Fact extraction (`extract_facts.py`, `src/website_pipeline/fact_rules.py`)

Every fact is either a **verbatim string** from a cleaned section, or a
mechanical join of verbatim strings (the "Our Locations" and testimonial
blocks list country/quote and attribution as separate lines with no
punctuation joining them - `fact_rules.py` groups them back into one
readable fact per country/testimonial using only the country name or the
name-title-dash-company pattern as the delimiter, never adding words).
Classified into `company / service / technology / location / contact /
portfolio / client / partner / industry / capability / company_value /
history / other` by URL path and heading keyword rules - see
`classify_fact_type()`.

Legal boilerplate pages (`/privacy-policy`, `/terms-conditions`,
`/cookie-policy`) are excluded by default (`--include-legal-pages` to
include them) - they're standard legal text, not FAQ-style company
information, and would otherwise dominate the fact bank with noise.

## 4. QA generation

**Auto-generated (`generate_qa.py`)**: Technyx's own `/faq` page and every
`/platforms/<name>` page publish real FAQ content (also present as
schema.org `FAQPage` JSON-LD). Where a fact's `source_section` is itself
phrased as a genuine question ("What Sitecore development services does
Technyx provide?"), turning it into a QA pair requires no paraphrasing -
the site's own answer *is* the fact text. A heading-must-start-with-an-
interrogative-word filter excludes CTA headings that coincidentally end in
"?" (e.g. "Ready to Build What's Next?").

**Hand-authored (`seed_curated_qa.py`, `seed_edge_case_qa.py`)**: everything
that requires actual judgment about phrasing, combining multiple facts, or
representing an absence of information. Every fact citation in these two
scripts is resolved by looking up the real fact text in `facts.jsonl`
(a `find(url_suffix, needle)` helper), not typed by hand, so a citation is
always to a fact that actually exists:

- `qa_curated.jsonl` - company/mission/vision/values, all 6 services, all 6
  industries, platforms/technology, locations, contact, client testimonials,
  multi-fact questions, portfolio, and 2 ambiguous-question examples. Every
  record's `fact_ids` were resolved by looking up the actual fact text in
  `facts.jsonl` (not typed by hand), so a citation is always to a real fact.
- `qa_followups.jsonl` - multi-turn conversations with `history`.
- `qa_unknown.jsonl` - topics checked against the full crawled corpus and
  confirmed **absent** before writing the record (pricing, employee count,
  named CEO, London office, opening hours, certifications, awards,
  money-back guarantees, revenue).
- `qa_out_of_scope.jsonl` - unrelated questions (weather, sports, coding
  help, general trivia).
- `qa_conversational.jsonl` - greetings/thanks.
- `qa_adversarial.jsonl` - **test-only** hallucination traps (asks about
  New York/Canada offices, blockchain/game-dev services, WordPress,
  ISO certification, awards, guarantees, named employees, Microsoft/AWS
  partnerships - none of which the site mentions), each labeled with an
  `expected_behavior` (`state_not_available`, `refuse_out_of_scope`, or
  `answer_from_source` for grounded control cases).

## 5. Validation (`validate_grounding.py`)

A mechanical lint pass (not a proof of correctness): every "grounded"
record must cite at least one real `fact_id`, its `source_url` must match
at least one cited fact's source (multi-fact answers may combine facts
from more than one page), and "not_available"/"out_of_scope"/
"conversational"/"clarification" records must not falsely cite facts. Also
flags exact duplicate (question, category) pairs.

## 6. Splitting (`build_datasets.py`)

- Adversarial/hallucination-trap examples are **test-only** by
  construction - never in train or validation.
- Every other record is grouped into a cluster with every other record that
  cites at least one of the same `fact_id` (via union-find), so paraphrases
  of the same underlying fact ("What services does Technyx provide?" /
  "What can Technyx help my business with?") always land in the same split.
  This is what avoids data leakage - it is **not** a plain random 90/10
  shuffle.
- Clusters are shuffled with a fixed seed (`--seed 42`, documented in
  `build_datasets.py`'s `--help`) and greedily assigned to hit an 80/10/10
  split by record count.
- The script then checks that every QA `category` present in the pool has
  at least one representative in the test set, moving a cluster from train
  if a category would otherwise be missing - so the test set deliberately
  covers direct/paraphrased/multi-part/follow-up/unknown/out-of-scope/
  hallucination-trap/ambiguous/contact/location/service/technology/
  portfolio questions, not just whatever a random sample happened to include.

## Known limitations (be aware of these before trusting the corpus blindly)

- The "Our Technology Ecosystem" tab widget on service/platform pages only
  server-renders its default "Frontend" tab (React, Next.js, Vue.js,
  Angular, TypeScript, Svelte, Tailwind, MUI); the other category tabs
  (Backend, Databases, AI/ML, Cloud, DevOps, Security, Search Engine,
  Creative) are populated client-side and were **not** captured - the
  pipeline does not guess what's in them.
- No individual employee names, a named CEO, specific project case studies,
  certifications, or awards are published anywhere on the site - this is
  reflected in `qa_unknown.jsonl`, not filled in with placeholders.
- Client/portfolio information is limited to what's named in the homepage's
  testimonial quotes (person, title, company) - there is no dedicated
  "our clients" or "case studies" page.
