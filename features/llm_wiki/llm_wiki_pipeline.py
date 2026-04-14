#!/usr/bin/env python3
"""
LLM Wiki Pipeline
=================

A full working implementation of the LLM Wiki pattern described in the documentation.

This script creates and maintains a persistent, LLM-driven Markdown wiki:
- raw/ : Immutable source files (text/markdown only in this MVP)
- wiki/ : LLM-maintained interlinked knowledge base
- schema.md : The "brain" — instructions the LLM follows strictly

Features implemented:
- init: Set up the project with directories and schema
- ingest: Process one source at a time → update 5–15 wiki pages intelligently
- query: Answer questions against the current wiki (with citations)
- lint: Health-check the wiki (contradictions, orphans, stale claims, etc.)

Dependencies (install once):
pip install openai pyyaml

Configuration:
- Set your API key as environment variable: export API_KEY=...
- Edit MODEL and API_BASE at the top of the script to use Grok (xAI), GPT-4o, Claude, etc.
  (xAI Grok API is fully supported and recommended)
"""

import argparse
import datetime
import json
import os
from pathlib import Path
from typing import Dict, List

from openai import OpenAI

# ========================= CONFIGURATION =========================
# Edit these to match your LLM provider
API_BASE = "https://api.x.ai/v1"  # xAI Grok (default)
MODEL = "grok-3"  # or "grok-beta", "grok-4", etc.
# For OpenAI fallback:
# API_BASE = "https://api.openai.com/v1"
# MODEL = "gpt-4o"

API_KEY_ENV = (
    "API_KEY"  # xAI uses XAI_API_KEY, OpenAI uses OPENAI_API_KEY — we alias both
)
# ================================================================

client = OpenAI(
    api_key=os.getenv(API_KEY_ENV)
    or os.getenv("XAI_API_KEY")
    or os.getenv("OPENAI_API_KEY"),
    base_url=API_BASE,
)

PROJECT_ROOT = Path(".")
RAW_DIR = PROJECT_ROOT / "raw"
WIKI_DIR = PROJECT_ROOT / "wiki"
SCHEMA_FILE = PROJECT_ROOT / "schema.md"

# Default schema (adapted directly from the provided documentation)
DEFAULT_SCHEMA = """# LLM Wiki Schema & Maintainer Instructions

You are the dedicated, tireless maintainer of this personal LLM Wiki — a persistent, compounding knowledge base.

## Core Philosophy (from the original documentation)
- The wiki sits between raw sources and the user. It is the single source of synthesized, interlinked knowledge.
- You never modify files in `raw/`. You only read them.
- Every new source is integrated: summaries written, entity/concept pages updated, contradictions flagged, cross-references added.
- The wiki compounds value over time. Maintenance cost is near zero because you do all the bookkeeping.

## Directory Structure (strict)
- `raw/`          → immutable source files (user drops them here)
- `wiki/`
  - `index.md`    → living catalog (organized by category, with one-line summaries + links)
  - `log.md`      → chronological append-only activity log
  - `sources/`    → one dedicated page per ingested source
  - `entities/`   → people, organizations, products, etc.
  - `concepts/`   → ideas, themes, theories, terms
  - `analyses/`   → filed query results, comparisons, syntheses
  - (You may create additional top-level folders only with clear justification)

## Every Wiki Page Format (MANDATORY)
```yaml
---
title: "Clear, descriptive title"
date: "YYYY-MM-DD"
tags: ["tag1", "tag2"]
sources: ["source-filename.md"]   # raw source references
---
```
Followed by rich Markdown with:
- Headings, bullet points, tables, **bold**, *italic*
- Obsidian-style internal links: [[Entity Name]] or [[sources/Article Title]]
- When new information contradicts old: "**Contradiction noted:** [new source] updates/challenges the claim on [[Old Page]]."

## index.md Structure
```markdown
# Wiki Index

## Overview
One-paragraph synthesis of the entire knowledge base.

## Sources
- [[sources/Source Title]] – one-line summary

## Entities
...

## Concepts
...
```
You must keep this file perfectly up-to-date on every ingest.

## log.md Structure
```markdown
# Wiki Activity Log

## [2026-04-14 15:30] ingest | Article Title
- Ingested raw/source.md
- Updated/created 12 pages
- Key integration: ...
```

## Ingest Workflow (you follow this exactly)
1. Read the new raw source.
2. First, output JSON plan: list of pages that must be touched.
3. Then (second call), receive current content of those pages and output full new Markdown for every affected page + full new index.md + new log entry.

## Query Workflow
- Review index.md → identify relevant pages
- Synthesize answer with inline [[citations]]
- If the answer is valuable, you may propose a filename in `analyses/` to file it permanently.

## Lint Workflow
Identify: contradictions, stale claims, orphan pages, missing cross-references, data gaps. Suggest next actions.

## General Rules
- Be objective, scholarly, and concise yet complete.
- Always integrate, never duplicate raw text.
- Use today's date for new/updated pages.
- Respond ONLY in the exact JSON format requested by the pipeline.
"""


def ensure_dirs():
    RAW_DIR.mkdir(exist_ok=True)
    WIKI_DIR.mkdir(exist_ok=True)
    (WIKI_DIR / "sources").mkdir(exist_ok=True)
    (WIKI_DIR / "entities").mkdir(exist_ok=True)
    (WIKI_DIR / "concepts").mkdir(exist_ok=True)
    (WIKI_DIR / "analyses").mkdir(exist_ok=True)


def load_file(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def save_file(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def call_llm(messages: List[Dict], json_mode: bool = False) -> str:
    kwargs = {"model": MODEL, "messages": messages, "temperature": 0.3}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content.strip()


def init():
    """One-time setup"""
    ensure_dirs()
    if not SCHEMA_FILE.exists():
        SCHEMA_FILE.write_text(DEFAULT_SCHEMA, encoding="utf-8")
        print("✅ Created schema.md (edit this to evolve your wiki conventions)")
    if not (WIKI_DIR / "index.md").exists():
        initial_index = """# Wiki Index

## Overview
Empty wiki — ready for your first ingest.

## Sources
## Entities
## Concepts
## Analyses
"""
        save_file(WIKI_DIR / "index.md", initial_index)
    if not (WIKI_DIR / "log.md").exists():
        save_file(WIKI_DIR / "log.md", "# Wiki Activity Log\n\n")
    print("✅ LLM Wiki initialized. Drop files into raw/ and run `ingest`.")


def get_current_index_and_log() -> tuple[str, str]:
    index = load_file(WIKI_DIR / "index.md")
    log = load_file(WIKI_DIR / "log.md")
    # Last 10 log entries for context
    recent_log = "\n".join(log.split("## [")[-10:])
    return (
        index,
        f"## [{datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}]"
        + recent_log.split("## [")[-1]
        if recent_log
        else "",
    )


def ingest(source_filename: str):
    """Full ingest pipeline"""
    source_path = RAW_DIR / source_filename
    if not source_path.exists():
        print(f"❌ Source not found: {source_path}")
        return

    source_content = load_file(source_path)
    schema = load_file(SCHEMA_FILE)
    index, recent_log = get_current_index_and_log()

    print(f"🔄 Ingesting {source_filename}...")

    # Step 1: Planning pass — which pages will be touched?
    planning_prompt = f"""You are the LLM Wiki maintainer. Follow the schema exactly.

Schema:
{schema}

Current index:
{index}

New source to ingest:
Filename: {source_filename}
Content:
{source_content}

Recent log:
{recent_log}

First, decide exactly which wiki pages must be created or updated.
Respond ONLY with valid JSON (no markdown, no extra text):

{{
  "affected_pages": [
    {{
      "page_path": "sources/my-source.md or entities/john-doe.md etc.",
      "action": "create or update",
      "reason": "one sentence"
    }}
  ]
}}
"""
    planning_response = call_llm(
        [
            {"role": "system", "content": "You are a precise JSON-only assistant."},
            {"role": "user", "content": planning_prompt},
        ],
        json_mode=True,
    )

    try:
        plan = json.loads(planning_response)
        affected = plan["affected_pages"]
        print(f"📋 Planning to touch {len(affected)} pages")
    except Exception as e:
        print("❌ Planning failed:", e)
        print(planning_response[:500])
        return

    # Step 2: Load current content of affected pages
    page_contents: Dict[str, str] = {}
    for item in affected:
        p = WIKI_DIR / item["page_path"]
        page_contents[item["page_path"]] = load_file(p)

    # Step 3: Generation pass — full new content for everything
    content_prompt = f"""You are the LLM Wiki maintainer. Follow the schema exactly.

Schema:
{schema}

Current index:
{index}

New source:
Filename: {source_filename}
Content:
{source_content}

Pages to update/create and their CURRENT content:
{json.dumps(page_contents, indent=2)}

Recent log:
{recent_log}

Now produce the COMPLETE updated wiki state.
Respond ONLY with valid JSON:

{{
  "updates": [
    {{
      "page_path": "sources/...",
      "full_markdown_content": "entire markdown including frontmatter..."
    }}
  ],
  "new_index": "complete new content of index.md",
  "log_entry": "full markdown block to APPEND to log.md (start with ## [date] ...)"
}}
"""
    generation_response = call_llm(
        [
            {"role": "system", "content": "You are a precise JSON-only assistant."},
            {"role": "user", "content": content_prompt},
        ],
        json_mode=True,
    )

    try:
        result = json.loads(generation_response)
        # Apply updates
        for update in result.get("updates", []):
            save_file(WIKI_DIR / update["page_path"], update["full_markdown_content"])
            print(
                f"   ✅ {'Created' if 'create' in [p['action'] for p in affected if p['page_path'] == update['page_path']][0] else 'Updated'} {update['page_path']}"
            )

        # Update index
        save_file(WIKI_DIR / "index.md", result["new_index"])
        print("   ✅ index.md updated")

        # Append log
        with open(WIKI_DIR / "log.md", "a", encoding="utf-8") as f:
            f.write("\n" + result["log_entry"] + "\n")
        print("   ✅ log.md updated")

        print(f"🎉 Ingest complete for {source_filename}")

    except Exception as e:
        print("❌ Generation failed:", e)
        print(generation_response[:1000])


def query(question: str, save_to_wiki: bool = False):
    """Answer a question against the current wiki"""
    schema = load_file(SCHEMA_FILE)
    index, _ = get_current_index_and_log()

    print(f"🤔 Answering: {question}")

    # Step 1: Find relevant pages
    relevance_prompt = f"""Schema:
{schema}

Current index:
{index}

Question: {question}

List the most relevant wiki pages (max 8).
Respond ONLY with JSON:
{{
  "relevant_pages": ["sources/xxx.md", "entities/yyy.md", ...]
}}
"""
    rel_response = call_llm(
        [
            {"role": "system", "content": "JSON-only assistant"},
            {"role": "user", "content": relevance_prompt},
        ],
        json_mode=True,
    )

    try:
        relevant = json.loads(rel_response)["relevant_pages"]
    except:
        relevant = []

    # Load their content
    context = ""
    for p in relevant:
        full_path = WIKI_DIR / p
        if full_path.exists():
            context += f"\n\n--- PAGE: {p} ---\n{load_file(full_path)}\n"

    # Step 2: Generate answer
    answer_prompt = f"""Schema:
{schema}

Relevant wiki content:
{context}

Question: {question}

Provide a clear, well-cited answer using [[page links]] where appropriate.
If the answer is substantial and worth keeping, also suggest a filename in analyses/ (e.g. analyses/why-x-matters.md)

Respond in normal Markdown (not JSON)."""

    answer = call_llm(
        [
            {"role": "system", "content": "You are a helpful, precise wiki assistant."},
            {"role": "user", "content": answer_prompt},
        ]
    )

    print("\n" + "=" * 80)
    print(answer)
    print("=" * 80)

    if save_to_wiki:
        # Auto-file the answer
        slug = question.lower().replace(" ", "-")[:60]
        path = WIKI_DIR / "analyses" / f"{slug}.md"
        content = f"""---
title: "Analysis: {question}"
date: "{datetime.date.today()}"
tags: ["analysis", "query"]
---

{answer}
"""
        save_file(path, content)
        print(f"💾 Saved to wiki: {path}")


def lint():
    """Run a health check on the wiki"""
    schema = load_file(SCHEMA_FILE)
    index, recent_log = get_current_index_and_log()

    print("🔍 Running lint...")

    lint_prompt = f"""Schema:
{schema}

Current index:
{index}

Recent log:
{recent_log}

Perform a full lint pass. Look for:
- Contradictions
- Stale claims
- Orphan pages
- Missing cross-references
- Important concepts without their own page
- Data gaps worth filling

Respond ONLY with JSON:
{{
  "issues": [
    {{
      "severity": "high/medium/low",
      "description": "...",
      "suggested_action": "ingest new source / update page / query ..."
    }}
  ],
  "summary": "one paragraph health summary"
}}
"""
    response = call_llm(
        [
            {"role": "system", "content": "JSON-only assistant"},
            {"role": "user", "content": lint_prompt},
        ],
        json_mode=True,
    )

    try:
        result = json.loads(response)
        print("\n📋 Lint Summary")
        print(result["summary"])
        print("\nIssues found:")
        for issue in result.get("issues", []):
            print(f"• [{issue['severity'].upper()}] {issue['description']}")
            print(f"  → Suggested: {issue['suggested_action']}")
    except Exception as e:
        print("Lint parsing failed:", e)


def main():
    parser = argparse.ArgumentParser(description="LLM Wiki Pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("init", help="Initialize the wiki project")

    ingest_p = subparsers.add_parser("ingest", help="Ingest a source from raw/")
    ingest_p.add_argument("source", help="Filename inside raw/ (e.g. article.md)")

    query_p = subparsers.add_parser("query", help="Ask a question against the wiki")
    query_p.add_argument("question", nargs="+")
    query_p.add_argument(
        "--save", action="store_true", help="Save answer as new analysis page"
    )

    subparsers.add_parser("lint", help="Run wiki health check")

    args = parser.parse_args()

    if args.command == "init":
        init()
    elif args.command == "ingest":
        ingest(args.source)
    elif args.command == "query":
        question = " ".join(args.question)
        query(question, save_to_wiki=args.save)
    elif args.command == "lint":
        lint()


if __name__ == "__main__":
    main()
