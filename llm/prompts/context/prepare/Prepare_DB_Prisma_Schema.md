You are given a user query, some textual context, all inside xml tags. You have to answer the query based on the context

<context>
{context}
</context>

<user_query>
Design a robust Prisma schema with postgres to define the application's DB schema based on all the features from context.

This schema should utilize Prisma features such as models, enums, constraints, and relationships to ensure a well-structured and maintainable database. Define models that represent entities in the context, establish relationships (e.g., one-to-many, many-to-many), and include appropriate field types, validations, and constraints.

Ensure the schema adheres to valid Prisma syntax, standards, supports scalability, and allows easy integration into various use cases outlined in the context.

Output only the Prisma schema wrapped in a code block (use ```prisma).
</user_query>
