import os
import re
from jet.llm.ollama.base import Ollama
from jet.logger import logger
from jet.transformers.formatters import format_json
from jet.llm.prompt_templates.base import generate_pydantic_models
from jet.file.utils import save_file

output_dir = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])

field_descriptions = """
A list of answers, each with anime title, document number and release year.
""".strip()
query = "Top otome villainess anime 2025"


def remove_imports(python_code):
    # Regular expression to match import lines
    pattern = r'^(import .*$|from .+ import .*$)'

    # Remove the lines that match the import pattern
    python_code_without_imports = re.sub(
        pattern, '', python_code, flags=re.MULTILINE)

    return python_code_without_imports.strip()


pydantic_models_context = f"Field Descriptions:\n{field_descriptions}\n\nQuery:\n{query}"
generated_python_code = generate_pydantic_models(
    context=pydantic_models_context)

schema_str = remove_imports(generated_python_code)

output_file = f"{output_dir}/generated_pydantic_models.py"
save_file(generated_python_code, output_file)
output_file = f"{output_dir}/generated_pydantic_schema.txt"
save_file(schema_str, output_file)
