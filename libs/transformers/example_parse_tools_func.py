import json
import os
import re
import shutil
import inspect
from typing import (
    Any,
    Dict,
    Callable,
    TypedDict,
    Literal,
    Optional,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)
from transformers.utils.chat_template_utils import (
    _parse_type_hint,
    parse_google_format_docstring,
    TypeHintParsingException,
    DocstringParsingException,
)
from jet.file.utils import save_file

OUTPUT_DIR = os.path.join(os.path.dirname(
    __file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)


def convert_type_hints_to_json_schema(func: Callable) -> dict:
    type_hints = get_type_hints(func)
    signature = inspect.signature(func)
    required = []
    for param_name, param in signature.parameters.items():
        if param.annotation == inspect.Parameter.empty:
            raise TypeHintParsingException(
                f"Argument {param.name} is missing a type hint in function {func.__name__}")
        if param.default == inspect.Parameter.empty:
            required.append(param_name)

    properties = {}
    for param_name, param_type in type_hints.items():
        properties[param_name] = _parse_type_hint(param_type)

    schema = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required

    return schema


def parse_docstring(func: Callable):
    doc = inspect.getdoc(func)
    if not doc:
        raise DocstringParsingException(
            f"Cannot generate JSON schema for {func.__name__} because it has no docstring!"
        )
    doc = doc.strip()
    main_doc, param_descriptions, return_doc = parse_google_format_docstring(
        doc)
    return main_doc, param_descriptions, return_doc


def validate_json_schema(json_schema, func):
    if (return_dict := json_schema["properties"].pop("return", None)) is not None:
        if return_doc is not None:  # We allow a missing return docstring since most templates ignore it
            return_dict["description"] = return_doc
    for arg, schema in json_schema["properties"].items():
        if arg not in param_descriptions:
            raise DocstringParsingException(
                f"Cannot generate JSON schema for {func.__name__} because the docstring has no description for the argument '{arg}'"
            )
        desc = param_descriptions[arg]
        enum_choices = re.search(
            r"\(choices:\s*(.*?)\)\s*$", desc, flags=re.IGNORECASE)
        if enum_choices:
            schema["enum"] = [c.strip()
                              for c in json.loads(enum_choices.group(1))]
            desc = enum_choices.string[: enum_choices.start()].strip()
        schema["description"] = desc

    output = {"name": func.__name__,
              "description": main_doc, "parameters": json_schema}
    if return_dict is not None:
        output["return"] = return_dict
    return {"type": "function", "function": output}


def get_current_weather(location: str, format: str):
    """
    Get the current weather

    Args:
        location: The city and state, e.g. San Francisco, CA
        format: The temperature unit to use. Infer this from the users location. (choices: ["celsius", "fahrenheit"])
    """
    pass


class WeatherResult(TypedDict):
    city: str
    temperature: int
    unit: str
    condition: str


def get_weather(city: str, unit: str) -> WeatherResult:
    """
    Get the weather for a specified city.

    Args:
        city: The city, e.g. San Francisco
        unit: The temperature unit to use. Infer this from the users location. (choices: ["celsius", "fahrenheit"])

    Returns:
        WeatherResult: A dictionary with the following fields:
            temperature: int, the temperature in the specified unit
            condition: str, the weather condition (e.g., 'Sunny')
    """
    return {"city": city, "temperature": 20, "unit": unit, "condition": "Sunny"}


def add_two_numbers(a: int, b: int) -> int:
    """
    Add two numbers

    Args:
        a: The first number
        b: The second number

    Returns:
        int: The sum of the two numbers
    """
    return int(a) + int(b)


json_schema = convert_type_hints_to_json_schema(get_current_weather)
save_file(json_schema, f"{OUTPUT_DIR}/get_current_weather_json_schema.json")
main_doc, param_descriptions, return_doc = parse_docstring(get_current_weather)
save_file({
    "main_doc": main_doc,
    "param_descriptions": param_descriptions,
    "return_doc": return_doc,
}, f"{OUTPUT_DIR}/get_current_weather_parsed_docstring.json")
result = validate_json_schema(json_schema, get_current_weather)
save_file(result, f"{OUTPUT_DIR}/get_current_weather_result.json")

json_schema = convert_type_hints_to_json_schema(get_weather)
save_file(json_schema, f"{OUTPUT_DIR}/get_weather_json_schema.json")
main_doc, param_descriptions, return_doc = parse_docstring(get_weather)
save_file({
    "main_doc": main_doc,
    "param_descriptions": param_descriptions,
    "return_doc": return_doc,
}, f"{OUTPUT_DIR}/get_weather_parsed_docstring.json")
result = validate_json_schema(json_schema, get_weather)
save_file(result, f"{OUTPUT_DIR}/get_weather_result.json")

json_schema = convert_type_hints_to_json_schema(add_two_numbers)
save_file(json_schema, f"{OUTPUT_DIR}/add_two_numbers_json_schema.json")
main_doc, param_descriptions, return_doc = parse_docstring(add_two_numbers)
save_file({
    "main_doc": main_doc,
    "param_descriptions": param_descriptions,
    "return_doc": return_doc,
}, f"{OUTPUT_DIR}/add_two_numbers_parsed_docstring.json")
result = validate_json_schema(json_schema, add_two_numbers)
save_file(result, f"{OUTPUT_DIR}/add_two_numbers_result.json")
