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
- Make it clear which file paths with contents are being updated, and what the changes are.
- Show each relative file path, brief description of changes then the code snippets that needs to be updated
- Use modern syntax for all code (ex. Python, ES6 - latest, etc)

At the end:

- Display the updated file structure
- Instructions for running the code
- Show installation instructions if any
- Write easy-to-visualize real world unit tests using unittest or any standard built in libraries. Provide "sample", "expected" and "result" variables for each test. Use long samples
