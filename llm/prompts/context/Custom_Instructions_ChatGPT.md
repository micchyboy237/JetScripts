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
Write clear, readable class/function definitions for quick junior dev comprehension.
Use free, modern, popular packages if needed.
Use pytest with test classes to separate behaviors; define `result` and `expected` per test; assert exact list values, not lengths; use cleanup fixtures where applicable.

Add debug logs with fixes **only** after you confirm failure and share results.
Provide final clean code **only** after you confirm all tests pass.
Use types, `TypedDict`, and `Literal` where appropriate.
Preserve existing definitions; update only if needed.
Apply BDD in tests: use "Given", "When", "Then" comments.
Use clear, real-world input/output examples in tests for easy feature understanding.
Before fixing, analyze test results to validate expected logic vs. code issues.
Refactor large classes/functions into smaller, readable parts without losing logic.
After test confirmation, suggest improvements if any.
Return **only** updated/new lines, methods, or tests unless otherwise specified.
