# Code Contexts

## Context 1

Dont use prior knowledge or add to memory.

{prompt}

## Context 2

Dont use prior knowledge or add to memory.

Show code goals and how to use.

{prompt}

## Context 3

SYSTEM
Dont use prior knowledge or add to memory.
Execute browse or internet search if requested.

INSTRUCTIONS

- Provide a step by step process of how you would solve the query.
- Keep the code short, readable, reusable, testable, maintainable, optimized and sophisticated. Follow best practices and industry design patterns.
- Install any libraries required to run the code.
- You may add files or update the code structure if necessary.
- Reuse existing code if possible without breaking anything.
- Only respond with parts of the code that have been added or updated to keep it short and concise.
- Make it clear which file paths with contents are being updated, and what the changes are.
- Show each relative file path, brief description of changes then the code snippets that needs to be updated.
- Include real world usage examples if applicable. Maintain existing args if provided.
- At the end, display the updated file structure and instructions for running the code.
- Ignore instructions that are not applicable to the query.

QUERY
{prompt}

## Context 4

Dont use prior knowledge or add to memory.

Refactor this code as classes with types and typed dicts for readability, modularity, and reusability.
Add main function for usage examples. Use existing params if exists.
At the end, show installation instructions if dependencies are provided.

{prompt}

## Context 5

System:
Dont use prior knowledge or add to memory.

Instructions:

- Analyze query.
- Generated code should be refactored for improved readability, modularity, and reusability.
- Add main function if it doesn't exist.
- Don't remove logs.

Query:
{prompt}

## Context 6

<!-- For describing notebook (.ipynb) to python (.py) code -->

Dont use prior knowledge or add to memory.

This python code is exported from .ipynb notebook file. Do the ff:

- Analyze the purpose of each usage example then provide real world use cases

{prompt}

## Context 7

<!-- For converting notebook (.ipynb) to python (.py) code -->

Dont use prior knowledge or add to memory.

This python code is exported from .ipynb notebook file. Do the ff:

- Refactor with functions by usage example.
- Add comments for each to explain the purpose.

{prompt}

## Context 8

<!-- For existing projects -->

Dont use prior knowledge or add to memory.
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

## Context 9

<!-- For creating projects -->

Dont use prior knowledge or add to memory.
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

## Context 10

Dont use prior knowledge or add to memory.

Update the my prompt user_query tag only, so that it asks to write prisma schema instead of json.
Rewrite parts to fully utilize the prisma entities such as enums, models, constraints, relationships, etc.

PROMPT:

{prompt}

## Context 11

Dont use prior knowledge or add to memory.

- Generate a json array of items with attributes (question: str, context: str) based from the doc below.
- This array should completely represent all available data in the doc.

Doc:
{prompt}

## Context 12

Dont use prior knowledge or add to memory.

Answer the query using the provided context information, and not prior knowledge

Context information from multiple sources is below:

{prompt}
