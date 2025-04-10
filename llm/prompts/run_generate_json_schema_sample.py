import os
from jet.file.utils import load_file
from jet.llm.prompt_templates.base import generate_json_schema, generate_json_schema_sample
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
        field_type = str  # Default type

        if field_schema.get("type") == "integer":
            field_type = int
        elif field_schema.get("type") == "string":
            field_type = str
        elif field_schema.get("type") == "array":
            # Handle array type
            items_schema = field_schema.get("items", {})
            if items_schema.get("type") == "string":
                field_type = list[str]
            elif items_schema.get("type") == "object":
                # Handle object inside array
                # Use a dictionary as a placeholder, you might want to create sub-models
                field_type = list[Dict[str, Any]]
        elif field_schema.get("type") == "number":
            field_type = float  # For float or number

        model_fields[field] = (field_type, ...)

    # Create and return the Pydantic model dynamically
    return create_model('DynamicModel', **model_fields)


if __name__ == "__main__":
    query = "Top otome villainess anime 2025"

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
