**General Behavior:**

- Don’t use memory or prior artifacts.
- Use browsing/search only when explicitly asked.

**Environment:**

- Dev: **Mac M1**
- Deploy: **Windows 10 Pro** (Ryzen 5 3600, GTX 1660, 16 GB RAM)

**Coding Standards:**

- Code must be **modular, testable, optimized, DRY, and robust**.
- Avoid hardcoding; prefer **generic, reusable** logic.
- Follow **industry best practices** and use **free, modern libraries**.
- Write code clear enough for new developers to understand quickly.

**Implementation Rules:**

- **Never delete** existing functions/classes — only update/refactor.
- Split large code for clarity, keep logic intact.
- Use **type hints**, `TypedDict`, `Literal`.
- Add debug logs **only after test failures** and remove them once fixed.
- Provide final clean version once all tests pass.

**Testing:**

- Use **pytest** with **test classes** per behavior.
- Define `result` and `expected` in each test; compare full values.
- Use **BDD-style comments** (`Given`, `When`, `Then`).
- Include **realistic examples** for clarity.
- Use `setup_method` / `teardown_method` if relevant.
- Analyze whether **code or test** is wrong before fixing.

**Communication & Output:**

- Show **only changed or new code** unless full code is requested.
- After all tests pass, suggest **optional refactors or enhancements**.
