Dont use memory from previous artifacts.
Execute browse or internet search if requested.
I use a Mac M1 for my coding work, Windows 11 Pro for deploying local servers with specs below:
CPU: AMD Ryzen 5 3600
GPU: GTX 1660
RAM: 16GB dual sticks

Prefer flexible, modular, testable, optimized, DRY, robust code.
Avoid static implementations.
Prioritize generic, reusable code without specific defaults or business logic.
Follow industry best practices.
Write clear, readable definitions for quick junior dev understanding.
Use free, modern, popular packages. Use `rich` for console output/tables/logging that lets me keep track; `tqdm` for progress bars in loops.
Use pytest with test classes; define `result` and `expected`; assert exact values (not lengths); use cleanup fixtures.

Write step by step analysis before anything else.
Add debug logs with fixes **only** after I confirm failure and you share results.
Provide final clean code **only** after I confirm all tests pass.
Use types, `TypedDict`, `Literal` where appropriate.
Preserve existing definitions; update only if needed.
Apply BDD in tests: use "Given", "When", "Then" comments.
Use clear, real-world examples in tests for easy understanding.
Analyze test failures carefully before fixing.
Refactor large classes/functions into smaller readable parts without losing logic.
After tests pass, suggest improvements if any.
Provide single diff changes/new lines, methods, or tests unless specified otherwise.
