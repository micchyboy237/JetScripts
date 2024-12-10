# Code Contexts

## Context 1

Dont use or add to memory.

{prompt}

## Context 2

Dont use or add to memory.

Show code goals and how to use.

{prompt}

## Context 3

Dont use or add to memory.

Refactor this code as classes with types and typed dicts for readability, modularity, and reusability.
Add main function for usage examples.
At the end, show installation instructions if dependencies are provided.

{prompt}

## Context 4

System:
Dont use or add to memory.

Instructions:

- Analyze query.
- Generated code should be refactored for improved readability, modularity, and reusability.
- Add main function if it doesn't exist.
- Don't remove logs.

Query:
{prompt}

## Context 5

<!-- For converting notebook (.ipynb) to python (.py) code -->

Dont use or add to memory.

Copy all python code from this notebook code.

{prompt}

## Context 6

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

## Context 7

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

## Context 8

Dont use or add to memory.

Update the my prompt user_query tag only, so that it asks to write prisma schema instead of json.
Rewrite parts to fully utilize the prisma entities such as enums, models, constraints, relationships, etc.

PROMPT:

{prompt}

## Context 9

Dont use or add to memory.

- Generate a json array of items with attributes (question: str, context: str) based from the doc below.
- This array should completely represent all available data in the doc.

Doc:
{prompt}
