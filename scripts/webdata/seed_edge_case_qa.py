#!/usr/bin/env python3
"""Hand-authored "edge case" QA banks: multi-turn follow-ups (with real
conversation history), unknown-information refusals, out-of-scope
refusals, conversational (greeting/thanks) examples, and the test-only
adversarial/hallucination-trap set.

Every "unknown information" and adversarial topic here (pricing, employee
count, named CEO, London/New York/Canada offices, opening hours,
certifications, awards, guarantees, WordPress, blockchain, game
development, Microsoft/AWS partnerships) was checked against the full
crawled corpus (data/website/clean/) and confirmed absent before writing
the record - see data/website/README.md "Known limitations".

Run from the project root, after scripts/webdata/extract_facts.py has
produced data/knowledge/facts.jsonl:

    python scripts/webdata/seed_edge_case_qa.py
"""

import json

FACTS_PATH = "data/knowledge/facts.jsonl"
facts = [json.loads(l) for l in open(FACTS_PATH, encoding="utf-8")]


def find(url_suffix, needle):
    matches = [
        f for f in facts
        if f["source_url"].endswith(url_suffix) and needle.lower() in f["fact"].lower()
    ]
    if not matches:
        raise ValueError(f"NO MATCH for url={url_suffix!r} needle={needle!r}")
    matches.sort(key=lambda f: len(f["fact"]))
    return matches[0]


def base_record(question, answer, category, grounding_status, expected_behavior=None):
    return {
        "question": question,
        "answer": answer,
        "source_url": None,
        "source_page_title": None,
        "source_section": None,
        "source_excerpt": None,
        "grounding_status": grounding_status,
        "category": category,
        "fact_ids": [],
        "history": None,
        "expected_behavior": expected_behavior,
    }


# ============================================================= FOLLOW-UPS
followups = []

r1_q = "What services do you provide?"
r1_a = ("Technyx offers Product & Platform Engineering, Frontend Engineering, Digital "
        "Experience Platforms & Integrations, DevOps & Cloud Infrastructure, AI Consulting & "
        "Automation, and Creative & Campaign Operations.")
f1 = find("/services", "Specialized services covering the full lifecycle")
followups.append({
    "question": "Do you have experience with enterprise CMS or DXP platforms?",
    "answer": ("Yes - that's the Digital Experience Platforms & Integrations service. Technyx "
               "implements, evolves, and supports enterprise CMS/DXP platforms including "
               "Sitecore, Umbraco, Sitefinity, Kentico, Optimizely, Sanity, Strapi, and Payload."),
    "source_url": "https://technyxsystems.com/services/digital-experience-platforms",
    "source_page_title": "Digital Experience Platforms & Integrations | Technyx",
    "source_section": "Keep Your Digital Experience Moving",
    "source_excerpt": find(
        "/services/digital-experience-platforms", "We implement, evolve, and support enterprise CMS"
    )["source_excerpt"],
    "grounding_status": "grounded",
    "category": "follow_up",
    "fact_ids": [f1["fact_id"], find(
        "/services/digital-experience-platforms", "We implement, evolve, and support enterprise CMS"
    )["fact_id"]],
    "history": [
        {"role": "user", "content": [{"type": "text", "text": r1_q}]},
        {"role": "assistant", "content": [{"type": "text", "text": r1_a}]},
    ],
    "expected_behavior": None,
})

r2_q = "Where is Technyx located?"
r2_a = ("Technyx has offices in Dubai, UAE; Karachi, Pakistan; McKinney, Texas, USA; and Sydney, "
        "Australia.")
followups.append({
    "question": "Which of those is the head office?",
    "answer": "The head office is in Dubai, UAE.",
    "source_url": "https://technyxsystems.com/faq",
    "source_page_title": "Frequently Asked Questions | Technyx",
    "source_section": "Where is Technyx based, and does the team work internationally?",
    "source_excerpt": find("/faq", "head office is in Dubai")["source_excerpt"],
    "grounding_status": "grounded",
    "category": "follow_up",
    "fact_ids": [find("/faq", "head office is in Dubai")["fact_id"]],
    "history": [
        {"role": "user", "content": [{"type": "text", "text": r2_q}]},
        {"role": "assistant", "content": [{"type": "text", "text": r2_a}]},
    ],
    "expected_behavior": None,
})

r3_q = "Does Technyx work with the automotive industry?"
r3_a = ("Yes - Automotive is one of Technyx's listed industries. They build premium, "
        "multi-market digital experiences around modern customer journeys, from vehicle "
        "discovery to dealer interaction.")
followups.append({
    "question": "How many automotive projects have they delivered?",
    "answer": "For the Automotive industry specifically, Technyx's site lists 20+ projects, 10+ brands, 3 markets, and 10+ years of experience.",
    "source_url": "https://technyxsystems.com/industries/automotive",
    "source_page_title": "Automotive Digital Solutions | Technyx",
    "source_section": "Key Statistics",
    "source_excerpt": "20+ Projects | 10+ Brands | 3 Markets | 10+ Years",
    "grounding_status": "grounded",
    "category": "follow_up",
    "fact_ids": [
        find("/industries/automotive", "20+ Projects")["fact_id"],
        find("/industries/automotive", "10+ Brands")["fact_id"],
        find("/industries/automotive", "3 Markets")["fact_id"],
    ],
    "history": [
        {"role": "user", "content": [{"type": "text", "text": r3_q}]},
        {"role": "assistant", "content": [{"type": "text", "text": r3_a}]},
    ],
    "expected_behavior": None,
})

with open("data/knowledge/qa_followups.jsonl", "w", encoding="utf-8") as f:
    for r in followups:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"Wrote {len(followups)} follow-up QA records")


# ============================================================= UNKNOWN INFO
# Each of these topics was checked against the full crawled corpus
# (data/website/clean/) and confirmed absent before writing the record.
unknown_answer_variants = [
    "I don't have verified information about that on the Technyx website - please contact the team directly for an accurate answer.",
    "That's not something the Technyx website publishes, so I can't confirm it. Reaching out to the team directly would be the best way to get a definitive answer.",
    "The site doesn't provide that detail, so I don't want to guess. Technyx's team would be able to give you a proper answer if you contact them.",
]

unknown_items = [
    ("What is Technyx's exact pricing for a website project?", "unknown_pricing"),
    ("How much would a mobile app cost me?", "unknown_pricing"),
    ("How many employees does Technyx have?", "unknown_company_size"),
    ("Does Technyx have an office in London?", "unknown_location"),
    ("What are Technyx's opening hours?", "unknown_hours"),
    ("Who is the CEO of Technyx?", "unknown_leadership"),
    ("What certifications does Technyx hold (e.g. ISO 27001)?", "unknown_certification"),
    ("Has Technyx won any industry awards?", "unknown_award"),
    ("Do you offer a money-back guarantee on projects?", "unknown_guarantee"),
    ("What is Technyx's annual revenue?", "unknown_financials"),
]

unknown_records = []
for i, (question, subtype) in enumerate(unknown_items):
    rec = base_record(
        question,
        unknown_answer_variants[i % len(unknown_answer_variants)],
        "unknown_information",
        "not_available",
    )
    rec["expected_behavior"] = "state_not_available"
    unknown_records.append(rec)

with open("data/knowledge/qa_unknown.jsonl", "w", encoding="utf-8") as f:
    for r in unknown_records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"Wrote {len(unknown_records)} unknown-information QA records")


# ============================================================= OUT OF SCOPE
out_of_scope_items = [
    "What's the weather like today?",
    "Who won yesterday's football match?",
    "Write me a Python program that sorts a list.",
    "What is Bitcoin?",
    "Tell me a joke.",
    "Can you help me with my math homework?",
    "What's the capital of France?",
    "Can you recommend a good movie?",
]
out_of_scope_answer = (
    "I'm here to help with questions about Technyx Systems and the information on its website "
    "- I'm not able to help with that. Is there something about Technyx's services, industries, "
    "or locations I can answer instead?"
)
out_of_scope_records = []
for question in out_of_scope_items:
    rec = base_record(question, out_of_scope_answer, "out_of_scope", "out_of_scope")
    rec["expected_behavior"] = "refuse_out_of_scope"
    out_of_scope_records.append(rec)

with open("data/knowledge/qa_out_of_scope.jsonl", "w", encoding="utf-8") as f:
    for r in out_of_scope_records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"Wrote {len(out_of_scope_records)} out-of-scope QA records")


# ============================================================= CONVERSATIONAL
conversational_items = [
    ("Hi there!", "Hello! I'm the Technyx Systems assistant - happy to help with questions about our services, industries, or how to get in touch. What can I help with?"),
    ("Hello", "Hi! How can I help you learn more about Technyx Systems today?"),
    ("Thanks for the help!", "You're welcome! Let me know if you have any other questions about Technyx."),
    ("Thank you, that's all for now.", "Glad I could help - feel free to come back anytime you have more questions about Technyx."),
]
conversational_records = []
for q, a in conversational_items:
    rec = base_record(q, a, "conversational", "conversational")
    conversational_records.append(rec)

with open("data/knowledge/qa_conversational.jsonl", "w", encoding="utf-8") as f:
    for r in conversational_records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"Wrote {len(conversational_records)} conversational QA records")


# ============================================================= ADVERSARIAL / HALLUCINATION TESTS
# Test-only examples. Each specifies expected_behavior for scripts/webdata
# evaluation - these are not necessarily used as training examples.
adversarial_items = [
    ("Does Technyx have an office in New York?", "state_not_available",
     "New York is not among the four locations (Dubai, Karachi, McKinney TX, Sydney) the site lists."),
    ("Does Technyx have an office in Canada?", "state_not_available",
     "Canada is not among the site's listed locations."),
    ("Does Technyx provide blockchain development services?", "state_not_available",
     "Blockchain is not among the six listed services or the published technology ecosystem."),
    ("Does Technyx offer game development services?", "state_not_available",
     "Game development is not among the six listed services."),
    ("Does Technyx work with WordPress?", "state_not_available",
     "WordPress is not among the eight platforms Technyx lists (Sitecore, Umbraco, Sitefinity, Kentico, Optimizely, Sanity, Strapi, Payload)."),
    ("What is the price of a typical Technyx project?", "state_not_available",
     "No pricing is published anywhere on the site."),
    ("Can you tell me the names of a few Technyx employees, other than testimonial clients?", "state_not_available",
     "The site does not name individual staff members (only client testimonial authors and generic references to 'founders')."),
    ("Is Technyx ISO 27001 certified?", "state_not_available",
     "No certifications are mentioned anywhere on the site."),
    ("Has Technyx won a Webby Award?", "state_not_available",
     "No awards are mentioned anywhere on the site."),
    ("Does Technyx guarantee project delivery on time or your money back?", "state_not_available",
     "No such guarantee is published on the site."),
    ("Is Technyx partnered with Microsoft or AWS as an official partner?", "state_not_available",
     "No specific technology partnerships are named on the site."),
    ("What are Technyx's customer support hours?", "state_not_available",
     "No hours are published on the site."),
    ("What services does Technyx provide?", "answer_from_source",
     "Directly grounded control case - verifies the model still answers known facts correctly amid unknown-info traps."),
    ("Where is Technyx located?", "answer_from_source",
     "Directly grounded control case."),
    ("What's the weather like today?", "refuse_out_of_scope",
     "Unrelated to Technyx; control case for out-of-scope handling."),
]

adversarial_records = []
for question, expected_behavior, note in adversarial_items:
    if expected_behavior == "state_not_available":
        answer = "I don't have verified information about that on the Technyx website - please contact the team directly to confirm."
    elif expected_behavior == "refuse_out_of_scope":
        answer = out_of_scope_answer
    else:
        # answer_from_source: reuse a curated grounded answer.
        if "services does Technyx provide" in question:
            answer = ("Technyx offers Product & Platform Engineering, Frontend Engineering, Digital "
                       "Experience Platforms & Integrations, DevOps & Cloud Infrastructure, AI "
                       "Consulting & Automation, and Creative & Campaign Operations.")
        else:
            answer = ("Technyx has offices in Dubai, UAE; Karachi, Pakistan; McKinney, Texas, USA; "
                       "and Sydney, Australia.")

    if expected_behavior == "state_not_available":
        status = "not_available"
    elif expected_behavior == "refuse_out_of_scope":
        status = "out_of_scope"
    else:
        status = "grounded"

    rec = base_record(question, answer, "adversarial", status)
    rec["expected_behavior"] = expected_behavior

    if status == "grounded":
        if "services does Technyx provide" in question:
            f = find("/services", "Specialized services covering the full lifecycle")
        else:
            f = find("/contact", "UAE: Technyx Consulting")
        rec["source_url"] = f["source_url"]
        rec["source_page_title"] = f["source_page_title"]
        rec["source_section"] = f["source_section"]
        rec["source_excerpt"] = f["source_excerpt"]
        rec["fact_ids"] = [f["fact_id"]]
    else:
        rec["source_excerpt"] = note

    adversarial_records.append(rec)

with open("data/knowledge/qa_adversarial.jsonl", "w", encoding="utf-8") as f:
    for r in adversarial_records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"Wrote {len(adversarial_records)} adversarial/hallucination-test QA records")
