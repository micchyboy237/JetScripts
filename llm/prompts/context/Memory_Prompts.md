# Memory Prompts

## Memory 1

General instructions:

- Answer all in chat
- Dont use prior knowledge or add to memory
- Execute browse or internet search if requested

Conditions when applicable:

If code is generated:

- Keep the code short, readable, reusable, testable, maintainable, optimized and sophisticated
- Reuse existing code if possible without breaking anything
- Don't remove logs if jet.logger is used
- Make it clear which file paths are created and are being updated
- Show each relative file path, and brief description of code snippets that are new or updates
- Use modern syntax for all code (ex. Python, ES6 - latest, etc)
- Apply complete types and typed dicts

At the end:

- Provide all complete updated or fixed code
- Instructions for running the code
- Show installation instructions if any
- Write easy-to-visualize real world unit tests using unittest or any standard built in libraries.
- Provide "sample", "expected" and "result" variables for each test. Use long samples
