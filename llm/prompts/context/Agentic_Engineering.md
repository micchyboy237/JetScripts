**Agentic engineering** is a modern approach (mainly from 2025–2026) to software development and AI system design that centers on **AI agents** — systems that can plan, use tools, take actions, and iterate toward a goal with limited human help.

### Core idea in software development

It is the disciplined practice of building software **with coding agents** (AI systems that both write _and execute_ code) instead of writing every line by hand or casually prompting an LLM.

- You give the agent a clear goal or specification.
- The agent works in a loop: plans → writes/edits code → runs it → checks results (tests, errors, etc.) → improves until the goal is met.
- You stay in control: you define requirements, provide tools and constraints, review output, and decide what ships.

Popular examples of coding agents include Claude Code, OpenAI Codex, and Gemini CLI.

**Simon Willison** (a well-known developer) defines it simply as: developing software with the assistance of coding agents that can write _and_ execute code. Code execution is the key difference — without it, LLM output is just text; with it, the agent can verify and improve real working software.

### How it differs from “vibe coding”

Andrej Karpathy (OpenAI co-founder) popularized both terms:

| Aspect        | Vibe Coding                     | Agentic Engineering                         |
| ------------- | ------------------------------- | ------------------------------------------- |
| Style         | Casual, “prompt and pray”       | Structured, professional                    |
| Human role    | Minimal review                  | Oversight, specs, validation, quality gates |
| Quality focus | Fast prototypes (often messy)   | Maintain professional standards             |
| Risk          | High technical debt / “AI slop” | Controlled via tests, reviews, guardrails   |

Agentic engineering keeps the quality bar high while letting agents handle much of the repetitive work.

### Broader meaning

Outside pure coding, **agentic engineering** also means designing systems of autonomous AI agents that can:

- Break big goals into steps
- Use tools (APIs, databases, code runners, browsers, etc.)
- Adapt based on results
- Coordinate with other agents or humans

Google Research describes it as treating LLMs as semi-autonomous systems that follow verifiable specifications, with a strong “harness” (tools + prompts + guardrails). Engineers focus more on debugging the _workflow_ than fixing individual lines of code.

### What humans still do

Writing code was never the whole job of an engineer. In agentic engineering you spend more time on:

- Deciding _what_ to build and why
- Writing clear specs and acceptance criteria
- Setting up tools, tests, and safety rails
- Reviewing and validating results
- Improving the agent’s instructions and environment over time

### Simple analogy

Traditional coding = laying every brick yourself.  
Agentic engineering = acting as the architect and construction manager: you design the building, set quality standards, and supervise the (AI) crew that does the actual construction.

In short: **agentic engineering** is the practical, quality-focused way of using powerful AI coding agents to build real software faster — while remaining responsible for the final result.
