1. Context Isolation and Multi-Agent Orchestration (Highly Popular)
   Divide tasks across specialized sub-agents, each with its own narrow, focused context window, coordinated by a lead/orchestrator agent. This effectively multiplies available context capacity through parallelism and separation of concerns.

Orchestrator-Worker Pattern (e.g., Anthropic's Research system): Lead agent plans, decomposes the query, spawns sub-agents for parallel subtasks (e.g., different research angles), and synthesizes results. Sub-agents operate independently with minimal shared context, then return condensed findings.
Benefits: Handles tasks exceeding single windows; ~90% performance gains in complex research; parallel tool use speeds things up dramatically.
Tips: Provide clear delegation instructions, task boundaries, output formats, and scaling rules (e.g., effort proportional to query complexity). Use shared external memory for high-level plans to maintain continuity.
