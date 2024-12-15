# Code Contexts

## Context 1

Dont use or add to memory.

{prompt}

## Context 2

Dont use or add to memory.

Show code goals and how to use.

{prompt}

## Context 3

SYSTEM
Dont use or add to memory.
Execute browse or internet search if requested.

INSTRUCTIONS

- Provide a step by step process of how you would solve the query.
- Keep the code short, reusable, testable, maintainable and optimized. Follow best practices and industry design patterns.
- Install any libraries required to run the code.
- You may update the code structure if necessary.
- Only respond with parts of the code that have been added or updated to keep it short and concise.
- Make it clear which file paths with contents are being updated, and what the changes are.
- Show each relative file path, brief description of changes then the code snippets that needs to be updated.
- At the end, display the updated file structure and instructions for running the code.
- Ignore instructions that are not applicable to the query.

QUERY
{prompt}

## Context 4

Dont use or add to memory.

Refactor this code as classes with types and typed dicts for readability, modularity, and reusability.
Add main function for usage examples. Use existing params if exists.
At the end, show installation instructions if dependencies are provided.

{prompt}

## Context 5

System:
Dont use or add to memory.

Instructions:

- Analyze query.
- Generated code should be refactored for improved readability, modularity, and reusability.
- Add main function if it doesn't exist.
- Don't remove logs.

Query:
{prompt}

## Context 6

<!-- For converting notebook (.ipynb) to python (.py) code -->

Dont use or add to memory.

Copy all python code from this notebook code.
Remove the commented code for brevity.

{prompt}

## Context 7

<!-- For existing projects -->

Dont use or add to memory.
Execute browse or search internet if requested.

Follow these if you are expected to provide code:

- Keep the code short, reusable, testable, maintainable and optimized.
- Follow best practices and industry design patterns.
- Install any libraries required to run the code.
- You may update the code structure if necessary.
- Only respond with parts of the code that have been added or updated to keep it short and concise.
- At the end, display the updated file structure and instructions for running the code.
- Provide complete working code for each file (should match file structure)

Query:
{prompt}

## Context 8

<!-- For creating projects -->

Dont use or add to memory.
Execute browse or search internet if requested.

Follow these if you are expected to provide code:

- Keep the code short, reusable, testable, maintainable and optimized.
- Follow best practices and industry design patterns.
- Install any libraries required to run the code.
- You may update the code structure if necessary.
- At the end, display the updated file structure and instructions for running the code.
- Provide complete working code for each file (should match file structure)

Query:
{prompt}

## Context 9

Dont use or add to memory.

Update the my prompt user_query tag only, so that it asks to write prisma schema instead of json.
Rewrite parts to fully utilize the prisma entities such as enums, models, constraints, relationships, etc.

PROMPT:

{prompt}

## Context 10

Dont use or add to memory.

- Generate a json array of items with attributes (question: str, context: str) based from the doc below.
- This array should completely represent all available data in the doc.

Doc:
{prompt}
