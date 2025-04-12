import os
from jet.data.base import create_dynamic_model
from jet.llm.prompt_templates.base import generate_json_schema, generate_json_schema_sample
from jet.validation.json_schema_validator import schema_validate_json
from pydantic import create_model, BaseModel
from typing import Any, Dict

output_dir = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])


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
