You are given a user query, some textual context, all inside xml tags. You have to answer the query based on the context

<context>
{context}
</context>

<user_query>
You are an advanced AI assistant. The context contains specific query requesting data. Your task is to generate a Pydantic model to structure the response.

- Analyze the query to determine the required fields and their data types.
- Include optional fields where applicable based on the query context.
- Ensure the Pydantic model is valid and adheres to Python conventions.

Output only the Pydantic model code wrapped in a code block (use ```python).
</user_query>
