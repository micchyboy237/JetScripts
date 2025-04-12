import json
from pydantic import BaseModel, create_model
from typing import Any, Dict, Optional, List, Type, Union
import os
from jet.file.utils import save_file
from jet.llm.prompt_templates.base import generate_browser_query_json_schema, generate_json_schema_sample
from jet.validation.json_schema_validator import schema_validate_json
from pydantic import create_model, BaseModel
from typing import Any, Dict
from jet.llm.prompt_templates.base import convert_json_schema_to_model_instance

output_dir = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])


if __name__ == "__main__":
    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])

    query = "TikTok online seller registration steps 2025"

    json_schema = generate_browser_query_json_schema(query)
    json_schema_sample = generate_json_schema_sample(json_schema, query)

    json_schema_file = f"{output_dir}/browser_query_json_schema.json"
    json_schema_sample_file = f"{output_dir}/browser_query_json_schema_sample.json"
    save_file(json_schema, json_schema_file)
    save_file(json_schema_sample, json_schema_sample_file)

    # Convert to model
    model_instance = convert_json_schema_to_model_instance(
        json_schema_sample, json_schema)
    print(model_instance)
