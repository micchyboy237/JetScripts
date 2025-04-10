import os
from jet.file.utils import save_file
from jet.llm.prompt_templates.base import generate_browser_query_json_schema, generate_json_schema_sample
from jet.validation.json_schema_validator import schema_validate_json
from pydantic import create_model, BaseModel
from typing import Any, Dict

output_dir = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])


def create_dynamic_model(schema: Dict[str, Any]) -> BaseModel:
    model_fields = {}

    # Extract properties from the schema
    properties = schema.get("properties", {})

    for field, field_schema in properties.items():
        # Map the field types in the schema to Pydantic types
        field_type = str
        if field_schema.get("type") == "integer":
            field_type = int
        elif field_schema.get("type") == "string":
            field_type = str

        # '...' indicates required field
        model_fields[field] = (field_type, ...)

    # Create the Pydantic model dynamically
    return create_model('DynamicModel', **model_fields)


if __name__ == "__main__":
    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])

    query = "Top otome villainess anime 2025"

    json_schema = generate_browser_query_json_schema(query)
    json_schema_sample = generate_json_schema_sample(json_schema, query)

    json_schema_file = f"{output_dir}/browser_query_json_schema.json"
    json_schema_sample_file = f"{output_dir}/browser_query_json_schema_sample.json"
    save_file(json_schema, json_schema_file)
    save_file(json_schema_sample, json_schema_sample_file)
