import os
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

pydantic_models_context = f"Field Descriptions:\n{field_descriptions}\n\nQuery:\n{query}"
generated_pydantic_models = generate_pydantic_models(
    context=pydantic_models_context)

logger.success(format_json(generated_pydantic_models))

pydantic_models_file = f"{output_dir}/generated_pydantic_models.py"
save_file(generated_pydantic_models, pydantic_models_file)
