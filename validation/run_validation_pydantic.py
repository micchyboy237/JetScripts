import json
from typing import Optional
from pydantic import BaseModel
from jet.validation import validate_json_pydantic
from jet.transformers import make_serializable
from jet.logger import logger


def main_invalid_json():
    logger.newline()
    logger.info("main_invalid_json()...")
    invalid_json_sample = """{
        "name: "John Doe",
        "age": 30
        "email": "johndoe@example.com"
    }"""
    result = validate_json_pydantic(invalid_json_sample, SampleModel)
    logger.success(json.dumps(make_serializable(result), indent=2))


def main_valid_json_incorrect():
    logger.newline()
    logger.info("main_valid_json_incorrect()...")
    valid_json_incorrect_sample = """{
        "name": 1000,
        "age": "30",
        "email": "johndoe@example.com"
    }"""
    result = validate_json_pydantic(valid_json_incorrect_sample, SampleModel)
    logger.success(json.dumps(make_serializable(result), indent=2))


def main_valid_json_correct():
    logger.newline()
    logger.info("main_valid_json_correct()...")
    valid_json_correct_sample = """{
        "name": "John Doe",
        "age": 30
    }"""
    result = validate_json_pydantic(valid_json_correct_sample, SampleModel)
    logger.success(json.dumps(make_serializable(result), indent=2))


def main_valid_dict_correct():
    logger.newline()
    logger.info("main_valid_json_correct()...")
    valid_dict_correct_sample = {
        "name": "John Doe",
        "age": 30,
        "email": "johndoe@example.com"
    }
    result = validate_json_pydantic(valid_dict_correct_sample, SampleModel)
    logger.success(json.dumps(make_serializable(result), indent=2))


class SampleModel(BaseModel):
    name: str
    age: int
    email: Optional[str] = None


if __name__ == "__main__":
    main_valid_dict_correct()
    main_valid_json_correct()
    main_valid_json_incorrect()
    main_invalid_json()
