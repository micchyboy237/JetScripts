import shutil
from pathlib import Path

from jet.data.base import create_dynamic_model
from jet.file.utils import save_file
from jet.llm.prompt_templates.llama_cpp import (
    generate_json_schema,
    generate_json_schema_sample,
)
from jet.validation.json_schema_validator import schema_validate_json

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    query = "Top villainess anime 2026"

    json_schema = generate_json_schema(query)
    json_schema_sample = generate_json_schema_sample(json_schema, query)

    # Validate generated sample with schema
    validation_result = schema_validate_json(json_schema_sample, json_schema)

    # Create the dynamic model based on the JSON schema
    DynamicModel = create_dynamic_model(json_schema)

    # Example JSON data (simulated as a Python dictionary)
    # json_data = {"name": "John", "age": 30}
    json_data = json_schema_sample

    # Create an instance of the dynamically created model
    model_instance = DynamicModel(**json_data)

    # Output the result
    print(model_instance)

    save_file(json_schema, OUTPUT_DIR / "json_schema.json")
    save_file(json_data, OUTPUT_DIR / "json_data.json")
