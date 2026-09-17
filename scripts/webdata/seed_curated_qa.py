#!/usr/bin/env python3
"""Hand-authored QA pairs that require actual judgment about phrasing,
paraphrasing, or combining more than one fact - company/mission/vision/
values, all 6 services, all 6 industries, platforms/technology, contact/
locations, client testimonials, multi-fact questions, portfolio, and
ambiguous-question examples.

This is deliberately NOT template-generated: unlike scripts/webdata/
generate_qa.py (which mechanically reuses the site's own FAQ Q&A with no
paraphrasing), natural multi-fact answers and paraphrases need a human (or
an LLM acting as one) to write well - but every citation below is resolved
against data/knowledge/facts.jsonl by find(), not typed by hand, so a
grounding claim always points at a real, existing fact. Re-run
scripts/webdata/validate_grounding.py after editing this file.

Run from the project root (paths below are relative to it), after
scripts/webdata/extract_facts.py has produced data/knowledge/facts.jsonl:

    python scripts/webdata/seed_curated_qa.py
"""

import json

FACTS_PATH = "data/knowledge/facts.jsonl"
OUT_PATH = "data/knowledge/qa_curated.jsonl"

facts = [json.loads(l) for l in open(FACTS_PATH, encoding="utf-8")]


def find(url_suffix, needle):
    matches = [
        f for f in facts
        if f["source_url"].endswith(url_suffix) and needle.lower() in f["fact"].lower()
    ]
    if not matches:
        raise ValueError(f"NO MATCH for url={url_suffix!r} needle={needle!r}")
    if len(matches) > 1:
        # Prefer the shortest matching fact (most specific / least noisy).
        matches.sort(key=lambda f: len(f["fact"]))
    return matches[0]


def qa(question, answer, category, lookups, grounding_status="grounded", history=None):
    """lookups: list of (url_suffix, needle) tuples."""
    resolved = [find(u, n) for u, n in lookups]
    return {
        "question": question,
        "answer": answer,
        "source_url": resolved[0]["source_url"],
        "source_page_title": resolved[0]["source_page_title"],
        "source_section": resolved[0]["source_section"],
        "source_excerpt": " | ".join(r["source_excerpt"] for r in resolved),
        "grounding_status": grounding_status,
        "category": category,
        "fact_ids": [r["fact_id"] for r in resolved],
        "history": history,
        "expected_behavior": None,
    }


records = []

# ---------------------------------------------------------------- company
records.append(qa(
    "Tell me about Technyx.",
    "Technyx is an AI-native digital engineering partner for agencies, brands, and startups. "
    "Engagements are senior-led, engineers work AI-augmented as part of everyday delivery, and "
    "the team stays close to the problem from architecture through production.",
    "company",
    [("/faq", "AI-native digital engineering partner for agencies")],
))
records.append(qa(
    "What does Technyx do?",
    "Technyx is a digital engineering partner that builds custom software, digital experience "
    "platforms, frontend experiences, cloud infrastructure, AI-driven automation, and ongoing "
    "creative/campaign operations for agencies, brands, and startups.",
    "company",
    [("/", "Technyx is an AI-native digital engineering partner")],
))
records.append(qa(
    "What makes Technyx different from a typical development agency?",
    "Every engagement is senior-led, meaning experienced engineers stay directly involved in "
    "technical decisions rather than sitting above a junior delivery team, and every engineer "
    "works AI-augmented to help move faster without sacrificing quality.",
    "company",
    [("/faq", "senior-led, meaning experienced engineers")],
))
records.append(qa(
    "What is Technyx's mission?",
    "Technyx's mission is to give agencies, brands, and startups access to senior-level digital "
    "engineering that takes ownership of the outcome, combining technical depth, AI-augmented "
    "delivery, and the discipline to get difficult work shipped right.",
    "company",
    [("/company", "To give agencies, brands, and startups access to senior-level")],
))
records.append(qa(
    "What is Technyx's vision?",
    "Technyx's vision is to build a technology company where technical excellence, ownership, "
    "and continuous innovation are the standard, translating better ways of working into better "
    "outcomes for the teams that trust them.",
    "company",
    [("/company", "To build a technology company where technical excellence")],
))
records.append(qa(
    "What values does Technyx work by?",
    "Technyx names three core values: \"Own the Outcome\" (taking responsibility beyond just "
    "following scope), \"Say What's True\" (being upfront about feasibility, timelines, and "
    "trade-offs), and \"Raise the Bar\" (holding a high technical standard regardless of "
    "engagement size).",
    "company_value",
    [
        ("/company", "We take responsibility for getting the work"),
        ("/company", "clear about feasibility, timelines, risks"),
        ("/company", "hold the work to a high technical standard"),
    ],
))
records.append(qa(
    "How long has Technyx been in business?",
    "Technyx started in 2014. Its journey has moved through Agency Partnerships, then Enterprise "
    "Platforms, and now AI-Native Engineering - more than a decade of delivery, grown mostly "
    "through returning clients, agency partners, and referrals rather than paid marketing.",
    "history",
    [("/faq", "We started in 2014")],
))
records.append(qa(
    "Where is Technyx headquartered, and does it work internationally?",
    "Technyx is based in four locations: its head office is in Dubai, UAE, alongside offices in "
    "Karachi, Pakistan, McKinney, Texas, USA, and Sydney, Australia.",
    "location",
    [("/faq", "head office is in Dubai")],
))
records.append(qa(
    "Who leads Technyx?",
    "Technyx's site says its approach comes from its founders, who combine academic insight with "
    "real-world technology leadership - it doesn't name specific individuals or title-holders "
    "(e.g. a named CEO) on the pages that are public.",
    "company",
    [("/faq", "Our approach comes from our founders")],
))
records.append(qa(
    "Can you give me some numbers about Technyx - clients, years, projects?",
    "By Technyx's own published figures: 30+ clients, 10+ years of delivery, 100+ projects "
    "delivered, and 4 markets served.",
    "company",
    [
        ("/", "30+ Clients"),
        ("/", "10+ Years of Delivery"),
        ("/", "100+ Projects Delivered"),
        ("/", "4 Markets Served"),
    ],
))
records.append(qa(
    "Does Technyx have any client testimonials?",
    "Yes. Technyx's site features testimonials from people including Shane Gorman (Head of "
    "Internal Communication at Nestlé), Tom Otton (Managing Director and Founder at Create), "
    "and Dina Saadeh (General Manager at Blue Barracuda), among others, describing Technyx as a "
    "reliable, committed delivery partner.",
    "client",
    [
        ("/", "Shane Gorman"),
        ("/", "Tom Otton"),
        ("/", "Dina Saadeh"),
    ],
))
records.append(qa(
    "Which companies has Technyx worked with?",
    "The testimonials published on Technyx's site name Nestlé, Create, Majid Al Futtaim, Blue "
    "Barracuda, FGS, THE AKKAAS, and SCPS. The site doesn't publish a full client list beyond "
    "these testimonial attributions.",
    "client",
    [
        ("/", "Nestlé"),
        ("/", "Majid Al Futtaim"),
        ("/", "Blue Barracuda"),
    ],
))

# ---------------------------------------------------------------- services
records.append(qa(
    "What services does Technyx provide?",
    "Technyx offers six core services: Product & Platform Engineering, Frontend Engineering, "
    "Digital Experience Platforms & Integrations, DevOps & Cloud Infrastructure, AI Consulting & "
    "Automation, and Creative & Campaign Operations.",
    "service",
    [("/services", "Specialized services covering the full lifecycle")],
))
records.append(qa(
    "What can Technyx help my business with?",
    "Technyx can help with custom software and platform engineering, frontend experience "
    "engineering, enterprise CMS/DXP implementation, cloud and DevOps infrastructure, AI "
    "consulting and automation, and ongoing creative/campaign production.",
    "service",
    [("/services", "Specialized services covering the full lifecycle")],
))
records.append(qa(
    "Does Technyx provide custom software development?",
    "Yes - that's their Product & Platform Engineering service: custom software, applications, "
    "and technical architecture built around how the business actually works, for cases where an "
    "off-the-shelf platform can't meet the need.",
    "service",
    [("/services/product-platform-engineering", "When an off-the-shelf platform cannot meet")],
))
records.append(qa(
    "Do you offer frontend development?",
    "Yes. Technyx's Frontend Engineering service builds high-performance, custom digital "
    "experiences for brands, campaigns, and platforms, for cases where visual quality and "
    "interaction can't be left to a template.",
    "service",
    [("/services/frontend-engineering", "We build custom frontend experiences for brands")],
))
records.append(qa(
    "Can Technyx build or manage a CMS / digital experience platform?",
    "Yes. Technyx's Digital Experience Platforms & Integrations service covers implementing, "
    "evolving, and supporting enterprise CMS and DXP platforms, including headless CMS "
    "experiences, so a platform keeps working well after its initial launch.",
    "service",
    [("/services/digital-experience-platforms", "We implement, evolve, and support enterprise CMS")],
))
records.append(qa(
    "Does Technyx handle DevOps and cloud infrastructure?",
    "Yes. Their DevOps & Cloud Infrastructure service covers cloud environments, deployment "
    "pipelines, automation, and infrastructure management to keep digital products reliable, "
    "secure, and able to scale.",
    "service",
    [("/services/devops-cloud-infrastructure", "We manage the cloud and infrastructure layer")],
))
records.append(qa(
    "What does Technyx's AI Consulting & Automation service involve?",
    "It's focused on practical AI: identifying where AI can create real value in a business's "
    "operations or customer experience, then building intelligent workflows or AI-enabled "
    "solutions - grounded in Technyx's own product-building experience rather than pure advisory.",
    "service",
    [("/services/ai-consulting-automation", "AI creates value when it is connected to a real business need")],
))
records.append(qa(
    "Do you offer ongoing creative or campaign support, not just one-off projects?",
    "Yes. Technyx's Creative & Campaign Operations service is an embedded creative team for "
    "ongoing content, campaigns, and communications, working as an extension of a client's team "
    "rather than a one-off production vendor.",
    "service",
    [("/services/creative-campaign-operations", "We operate as an extension of your team")],
))

# ---------------------------------------------------------------- industries
records.append(qa(
    "What industries does Technyx work with?",
    "Technyx works across six named industries: Government & Public Sector, Tourism, Culture & "
    "Entertainment, Retail, E-commerce & FMCG, Banking, Financial Services & Fintech, Automotive, "
    "and Real Estate & Property.",
    "industry",
    [("/industries", "Premium, multi-market")],
))
records.append(qa(
    "Does Technyx work with automotive companies?",
    "Yes - Technyx lists Automotive as one of its industries, building premium, multi-market "
    "digital experiences around modern customer journeys, from vehicle discovery to dealer "
    "interaction.",
    "industry",
    [("/industries/automotive", "From vehicle discovery to campaign launches")],
))
records.append(qa(
    "Do you have experience in banking or financial services?",
    "Yes, Banking, Financial Services & Fintech is one of Technyx's listed industries - they "
    "describe building trusted digital experiences and products for complex financial "
    "environments.",
    "industry",
    [("/industries", "Trusted digital experiences")],
))
records.append(qa(
    "Does Technyx build for the retail or e-commerce sector?",
    "Yes. Retail, E-commerce & FMCG is one of Technyx's listed industries, focused on commerce "
    "experiences and content operations for always-on consumer brands.",
    "industry",
    [("/industries/retail-ecommerce-fmcg", "Retail, E-commerce & FMCG")],
))
records.append(qa(
    "Do you work with government or public sector organizations?",
    "Yes, Government & Public Sector is one of Technyx's listed industries - they describe "
    "building digital platforms around public impact, complex ecosystems, and long-term "
    "continuity.",
    "industry",
    [("/industries", "Digital platforms built around public impact")],
))
records.append(qa(
    "What about tourism or entertainment - does Technyx work in that space?",
    "Yes, Tourism, Culture & Entertainment is one of Technyx's listed industries: experience-led "
    "platforms connecting inspiration, discovery, planning, and visitor journeys.",
    "industry",
    [("/industries", "Experience-led platforms")],
))
records.append(qa(
    "Does Technyx do work in real estate?",
    "Yes, Real Estate & Property is one of Technyx's listed industries - focused on property "
    "experiences that connect storytelling, discovery, and lead journeys.",
    "industry",
    [("/industries", "Property experiences")],
))

# ---------------------------------------------------------------- platforms / technology
records.append(qa(
    "What content management systems or platforms does Technyx work with?",
    "Technyx works with Sitecore, Umbraco, Sitefinity, Kentico, Optimizely, Sanity, Strapi, and "
    "Payload.",
    "technology",
    [("/company", "Enterprise DXP for complex, multi-site experiences")],
))
records.append(qa(
    "Does Technyx work with Sitecore?",
    "Yes. Technyx provides end-to-end Sitecore development, including solution architecture, "
    "Sitecore XP/XM and XM Cloud implementations, custom component development, headless "
    "delivery with JSS/Next.js, integrations, upgrades, migrations, and ongoing support.",
    "technology",
    [("/platforms/sitecore", "Technyx provides end-to-end Sitecore development services")],
))
records.append(qa(
    "What frontend technologies does Technyx use?",
    "On their published technology ecosystem, Technyx lists React, Next.js, Vue.js, Angular, "
    "TypeScript, Svelte, Tailwind, and MUI under Frontend. (Their site organizes the wider stack "
    "into categories like Backend, Databases, AI/ML, Cloud, and DevOps too, but only the Frontend "
    "list is published in detail.)",
    "technology",
    [("/", "React")],
))
records.append(qa(
    "Is Technyx tied to one specific technology stack?",
    "No - Technyx describes itself as technology-agnostic by design, choosing technology around "
    "the requirement rather than a fixed stack, across enterprise DXPs, headless platforms, "
    "modern frontend/backend, cloud, and AI.",
    "technology",
    [("/", "We’re technology-agnostic by design")],
))

# ---------------------------------------------------------------- contact / locations
records.append(qa(
    "How can I contact Technyx?",
    "You can email info@technyxsystems.com, connect on WhatsApp, or use the contact form on "
    "their website.",
    "contact",
    [("/contact", "info@technyxsystems.com"), ("/contact", "Connect on WhatsApp")],
))
records.append(qa(
    "Where is Technyx located?",
    "Technyx has offices in four locations: Dubai, UAE (Technyx Consulting, IFZA Business Park, "
    "DDP); Karachi, Pakistan (Technyx Systems, B-38, Block 4, Gulshan-e-Iqbal); McKinney, Texas, "
    "USA (5701 Sidney Lane); and Sydney, Australia (Level 24, 3 International Towers, 300 "
    "Barangaroo Avenue, Sydney NSW 2000).",
    "location",
    [
        ("/contact", "UAE: Technyx Consulting"),
        ("/contact", "Pakistan: Technyx Systems"),
        ("/contact", "USA: McKinney"),
        ("/contact", "Australia: Level 24"),
    ],
))
records.append(qa(
    "Do you have an office in Pakistan?",
    "Yes - Technyx Systems has an office in Karachi, Pakistan, at B-38, Block 4, "
    "Gulshan-e-Iqbal.",
    "location",
    [("/contact", "Pakistan: Technyx Systems")],
))
records.append(qa(
    "Is Technyx based in the US?",
    "Yes, Technyx has a US location in McKinney, Texas (5701 Sidney Lane).",
    "location",
    [("/contact", "USA: McKinney")],
))
records.append(qa(
    "Do you have a presence in Australia?",
    "Yes - Technyx has an office in Sydney, Australia, at Level 24, 3 International Towers, 300 "
    "Barangaroo Avenue, Sydney NSW 2000.",
    "location",
    [("/contact", "Australia: Level 24")],
))

# ---------------------------------------------------------------- agencies/brands/startups
records.append(qa(
    "Do you work with agencies, or only directly with brands?",
    "Both. For agencies, Technyx works behind the scenes as a white-label technical partner - "
    "bringing senior engineering and confidentiality while the agency keeps the client "
    "relationship. For brands, they work as an extension of the internal team, and for startups "
    "they take ownership of the technical path from architecture through to launch.",
    "capability",
    [
        ("/", "white-label technical partner"),
        ("/", "extension of your digital team"),
        ("/", "Bring the product vision"),
    ],
))
records.append(qa(
    "I run a startup with just an idea - can Technyx help?",
    "Yes - Technyx's startup offering is built around exactly that: you bring the product vision, "
    "and they take ownership of the technical path from architecture and first build through "
    "infrastructure, integrations, and continuous evolution.",
    "capability",
    [("/startups", "turn your product vision into a working digital product")],
))

# ---------------------------------------------------------------- multi-fact
records.append(qa(
    "What services does Technyx provide, and how can I get in touch with them?",
    "Technyx offers Product & Platform Engineering, Frontend Engineering, Digital Experience "
    "Platforms & Integrations, DevOps & Cloud Infrastructure, AI Consulting & Automation, and "
    "Creative & Campaign Operations. You can reach them by email at info@technyxsystems.com, "
    "via WhatsApp, or through the contact form on their site.",
    "multi_fact",
    [
        ("/services", "Specialized services covering the full lifecycle"),
        ("/contact", "info@technyxsystems.com"),
    ],
))
records.append(qa(
    "Where is Technyx based, and how long have they been operating?",
    "Technyx is headquartered in Dubai, UAE, with additional offices in Karachi (Pakistan), "
    "McKinney, Texas (USA), and Sydney (Australia). They've been operating since 2014 - more "
    "than a decade of delivery.",
    "multi_fact",
    [
        ("/faq", "head office is in Dubai"),
        ("/faq", "We started in 2014"),
    ],
))

# ---------------------------------------------------------------- portfolio
records.append(qa(
    "Can you share examples of Technyx's past work or projects?",
    "Technyx doesn't publish individual project case studies on its site. What is published: "
    "client testimonials (e.g. from Nestlé, Create, Majid Al Futtaim, Blue Barracuda, FGS, "
    "THE AKKAAS, and SCPS) and the headline figure of 100+ projects delivered over 10+ years.",
    "portfolio",
    [
        ("/", "Nestlé"),
        ("/", "100+ Projects Delivered"),
    ],
))

# ---------------------------------------------------------------- ambiguous
records.append({
    "question": "Do you support that?",
    "answer": "I want to make sure I answer the right question - could you tell me a bit more "
              "about what you're referring to (a specific service, platform, or something else)?",
    "source_url": None,
    "source_page_title": None,
    "source_section": None,
    "source_excerpt": None,
    "grounding_status": "clarification",
    "category": "ambiguous",
    "fact_ids": [],
    "history": None,
    "expected_behavior": None,
})
records.append({
    "question": "What about the other one?",
    "answer": "I'm not sure which one you mean yet - could you clarify what you're comparing or "
              "asking about (e.g. a specific service, platform, or industry)?",
    "source_url": None,
    "source_page_title": None,
    "source_section": None,
    "source_excerpt": None,
    "grounding_status": "clarification",
    "category": "ambiguous",
    "fact_ids": [],
    "history": None,
    "expected_behavior": None,
})

write_count = len(records)
with open(OUT_PATH, "w", encoding="utf-8") as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"Wrote {write_count} curated QA records to {OUT_PATH}")
