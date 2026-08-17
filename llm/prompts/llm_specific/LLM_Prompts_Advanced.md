Please answer the following questions about yourself as completely and accurately as possible. Clearly separate **confirmed public information** from inference or environment-specific details. If something is unknown or restricted, say so directly. Prefer precision over marketing language.

**1. Identity**

- Official name / model identifier
- Developer
- Exact version or variant (if known)
- Primary design goals
- Maximum context length / native context window

**2. Knowledge cutoff & available tools**

- Most precise knowledge cutoff date (year + month if known). If only a year is known, state that clearly.
- List every tool available in this environment with a one-line description.
- Main limitations of each major tool (especially search, fetch, code execution, and any multimodal tools).

**3. Core capabilities & limitations**
Provide a table for these categories:

- Reasoning & analysis
- Coding & software engineering
- Creativity & generation
- Tool use / browsing / code execution
- Multimodal abilities (text, image, video, audio)
- Safety / refusal behavior

For each category: strengths + known limitations.  
Explicitly state whether you can process **uploaded** images, video, or audio **natively** (yes/no + brief note).

**4. Architecture, scale, context & memory management**

- Publicly confirmed architecture details only (dense/MoE, parameters, etc.). Mark everything else as unknown or speculation.
- Native context window and practical effective context (if different).
- How you manage long conversations, token limits, and memory (e.g., truncation, summarization, none).

**5. Decision logic & high-level approach style**

- Do you rewrite, expand, or decompose queries before acting? Briefly describe.
- Clear criteria for answering from knowledge vs calling tools.
- High-level approach style (ReAct-style, plan-then-act, agentic loop, or other).

**6. Operational workflows**
For each scenario, give concrete steps + a simple Mermaid diagram + key adaptations:

a) Factual question that may need current or post-cutoff information  
b) Writing or debugging non-trivial code  
c) Complex multi-part problem → structured recommendation

**7. Verification example**

- One short, concrete example of a recent capability improvement in your model family.
- Brief real-world example of a typical tool call **or** native image-handling step (whichever applies to you).

Answer in a clean, structured format.
