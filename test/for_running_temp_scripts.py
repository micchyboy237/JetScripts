import json
from jet.llm.prompt_templates.base import generate_json_schema
from pydantic import create_model, BaseModel
from typing import Any, Dict

# JSON schema (you can replace this with the schema you're working with)
json_schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"}
    },
    "required": ["name", "age"]
}

# Function to create a dynamic Pydantic model from a JSON schema


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
    query = "Top otome villainess anime 2025"

    json_schema_context = f"Query:\n{query}"
    json_schema = generate_json_schema(context=json_schema_context)

    # Create the dynamic model based on the JSON schema
    DynamicModel = create_dynamic_model(json_schema)

    # Example JSON data (simulated as a Python dictionary)
    json_data = {"name": "John", "age": 30}

    # Create an instance of the dynamically created model
    model_instance = DynamicModel(**json_data)

    # Output the result
    print(model_instance)
