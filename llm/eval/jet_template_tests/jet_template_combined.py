import os

base_deps_dir = "/Users/jethroestrada/Desktop/External_Projects/jet_python_modules"
deps_file_paths = [
    f"{base_deps_dir}/jet/logger/logger.py",
    f"{base_deps_dir}/jet/transformers/object.py",
    f"{base_deps_dir}/jet/llm/ollama/config.py",
]
orig_template_path = "/Users/jethroestrada/Desktop/External_Projects/JetScripts/llm/eval/jet_template_improved_sample.py"
improved_template_path = "/Users/jethroestrada/Desktop/External_Projects/JetScripts/llm/eval/jet_template_improved_sample.py"

file_contents = {
    "deps": {},
    "orig_template": "",
    "improved_template": "",
}

for path in deps_file_paths:
    with open(path, 'r') as file:
        file_contents['deps'][path] = file.read()

with open(orig_template_path, 'r') as file:
    file_contents['orig_template'] = file.read()

with open(improved_template_path, 'r') as file:
    file_contents['improved_template'] = file.read()

dependencies = []
for file_path, content in file_contents['deps'].items():
    rel_path = os.path.relpath(file_path, start=base_deps_dir)
    dependencies.append(f"File: {rel_path}\nContent:\n{content}")

dependencies_str = "\n".join(dependencies).strip()
old_code = file_contents['orig_template'].strip()
improved_code = file_contents['improved_template'].strip()

PROMPT_TEMPLATE = """
Prompt:
{prompt}
```python
{code}
```

Response:
""".strip()

CONTEXT_TEMPLATE = """
Sample prompt and response:
Prompt:
{prompt}
```python
{code}
```

Response:
```python
{response}
```
""".strip()

CONTEXT = CONTEXT_TEMPLATE.format(
    # dependencies=dependencies_str,
    prompt="Improve this code",
    code=old_code,
    response=improved_code,
)

IMPROVE_NOTEBOOK_TEMPLATE = """
Use the following context as your learned knowledge, inside <context></context> XML tags.
<context>
    {context}
</context>

When answer to user:
- If you don't know, just say that you don't know.
- If you don't know when you are not sure, ask for clarification.
Avoid mentioning that you obtained the information from the context.
And answer according to the language of the user's question.

Given the context information, answer the query.
Query: {query}
""".strip()


def generate_improve_prompt(code, prompt="Improve this code", context=CONTEXT):
    return IMPROVE_NOTEBOOK_TEMPLATE.format(
        context=context,
        query=PROMPT_TEMPLATE.format(
            prompt=prompt,
            code=code,
        )
    )
