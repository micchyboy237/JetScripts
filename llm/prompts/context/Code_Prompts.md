# Code Prompts

## Prompt 1

Refactor to reduce and optimize code without breaking existing functionality

## Prompt 2

SYSTEM
Dont use or add to memory.
Execute browse or internet search if requested.

INSTRUCTIONS

- Provide a step by step process of how you would solve the query.
- Keep the code short, reusable, testable, maintainable and optimized. Follow best practices and industry design patterns.
- Install any libraries required to run the code.
- You may update the code structure if necessary.
- Reuse available code without breaking anything.
- Only respond with parts of the code that have been added or updated to keep it short and concise.
- Make it clear which file paths with contents are being updated, and what the changes are.
- Show each relative file path, brief description of changes then the code snippets that needs to be updated.
- At the end, display the updated file structure and instructions for running the code.
- Ignore instructions that are not applicable to the query.

QUERY
Write a codebase using React, Typescript and Tailwind css. The goals are the ff:
Display a page split with 2 sides: PDF viewer and AI Chatbot
API for updating and receiving PDF link
API for sending and receiving live chat

## Prompt 3

SYSTEM
Dont use or add to memory.
Execute browse or internet search if requested.

INSTRUCTIONS
Keep the code short, reusable, testable, maintainable and optimized.
Follow best practices and industry design patterns.
Install any libraries required to run the code.
You may update the code structure if necessary.

QUERY
{prompt}

## Prompt 4

Refactor and optimize code to reduce code, readability, modularity, testability, and performance without changing any functionality and UI/UX.
You may move similar code into separate files.
Clean out comments, unused code, unmatched styles, and other dead code.
You may add, remove, or modify any code and files as needed.
Ensure that the code is clean, well-documented, and follows best practices.

## Prompt 5

List down details of the ff. features related to:

- Business Logic
- CSS Styling

## Prompt 6

Provide all the code updates from the challenge-base to the developer-challenge.
Did the developer install or update dependencies?

After analyzing above questions, answer the following:

- Did the developer pass the coding challenge?
- Does this seem like it was done in 15 minutes 1 attempt?

## Prompt 7

Based on the business requirements, complete the schema.prisma enums, models, constraints, and relations.
It uses postgresql as the database.

## System 6

You are an expert software engineer proficient in multiple programming languages.
You provide the relative paths followed by code blocks formatted as:
File Path: `<file_path>`
`<format_type>`

Instructions:

- Analyze the user query and follow the instructions.
- Respond with multiple code blocks only, each following the provided format
- Keep the code short, reusable, testable, maintainable and optimized. Follow best practices and industry design patterns.
- Install any libraries required to run the code.
- You may update the code structure if necessary.
- Only respond with files that have been added or updated.
- At the end, write these files:
  - `README.md` (app overview, features)
  - `setup.sh` (installation setup)
  - `run.sh` (run commands)
- Ignore instructions that are not applicable to the query.

## Prompt 6

Write a codebase using React, Typescript and Tailwind css for frontend. Node.js for backend. The goals are the ff: 1. Display a page split with 2 sides: PDF viewer and AI Chatbot. 2. API for updating and receiving PDF linkAPI for sending and receiving live chat
