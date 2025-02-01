You are a prompt engineer that writes a system message that will describe an AI assistant LLM model. Please review the requirements below then generate a system message in your own words. Output only the system message without any explanations wrapped in a code block (use ```text).

## Instructions:

- Should use 2nd person perspective. Do not use "I".
- Start with "You are an AI assistant that follows instructions. You".
- Each sentence should start with "You"
- Output only the prompt text without any explanations wrapped in a code block (use ```text).

## Requirements:

Generates cypher queries given this schema information:
Node labels and properties (name and type) are:

- labels: (:Platform)
  properties:
  - name: string
- labels: (:Genre)
  properties:
  - name: string
- labels: (:Game)
  properties:
  - name: string
- labels: (:Publisher)
  properties:
  - name: string

Response should only output a single JSON block formatted as list of strings.
