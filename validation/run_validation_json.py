import json
from jet.validation import validate_json
from jet.transformers import make_serializable
from jet.logger import logger


def main_invalid_json():
    logger.newline()
    logger.info("main_invalid_json()...")
    resume_schema = get_resume_schema()
    invalid_json_sample = """{
        "scope_of_work": ["backend", "web"],
        "job_description": "Develop and maintain web applications.",
        "qualifications": ["5+ years experience", "Proficient in Python"],
        "responsibilities": ["Web development", "Backend development"],
        "tech_stack": {
            "frontend": ["React", "Vue.js"],
            "backend": ["Node.js", "Python"],
            "database": ["PostgreSQL", "MongoDB"],
            "other": ["AWS"]
        },
        "level": "expert",  # Invalid value, should be one of ["entry", "mid", "senior"]
        "salary": {
            "min": 900,
            "max": 1100,
            "currency": "USD",
            "period": "monthly"
        }
    }"""
    result = validate_json(invalid_json_sample, resume_schema)
    logger.success(json.dumps(make_serializable(result), indent=2))


def main_valid_json_incorrect():
    logger.newline()
    logger.info("main_valid_json_incorrect()...")
    resume_schema = get_resume_schema()
    valid_json_incorrect_sample = """{
        "scope_of_work": ["backen", "web"],
        "job_description": "Develop and maintain web applications.",
        "qualifications": ["5+ years experience", "Proficient in Python"],
        "responsibilities": ["Web development", "Backend development"],
        "tech_stack": {
            "frontend": ["React", "Vue.js"],
            "backend": ["Node.js", "Python"],
            "database": ["PostgreSQL", "MongoDB"],
            "other": ["AWS"]
        },
        "level": "Mid-Senior level",
        "salary": {
            "min": 900,
            "max": "1100",
            "currency": "Php",
            "period": "Month"
        }
    }"""
    result = validate_json(valid_json_incorrect_sample, resume_schema)
    logger.success(json.dumps(make_serializable(result), indent=2))


def main_valid_json_correct():
    logger.newline()
    logger.info("main_valid_json_correct()...")
    resume_schema = get_resume_schema()
    valid_json_correct_sample = """{
        "scope_of_work": ["backend", "web"],
        "job_description": "Develop and maintain web applications.",
        "qualifications": ["5+ years experience", "Proficient in Python"],
        "responsibilities": ["Web development", "Backend development"],
        "tech_stack": {
            "frontend": ["React", "Vue.js"],
            "backend": ["Node.js", "Python"],
            "database": ["PostgreSQL", "MongoDB"],
            "other": ["AWS"]
        },
        "level": "mid",
        "salary": {
            "min": 900,
            "max": 1100,
            "currency": "USD",
            "period": "monthly"
        }
    }"""
    result = validate_json(valid_json_correct_sample, resume_schema)
    logger.success(json.dumps(make_serializable(result), indent=2))


def main_valid_dict_correct():
    logger.newline()
    logger.info("main_valid_dict_correct()...")
    resume_schema = get_resume_schema()
    valid_dict_correct_sample = {
        "scope_of_work": ["backend", "web"],
        "job_description": "Develop and maintain web applications.",
        "qualifications": ["5+ years experience", "Proficient in Python"],
        "responsibilities": ["Web development", "Backend development"],
        "tech_stack": {
            "frontend": ["React", "Vue.js"],
            "backend": ["Node.js", "Python"],
            "database": ["PostgreSQL", "MongoDB"],
            "other": ["AWS"]
        },
        "level": "mid",
        "salary": {
            "min": 900,
            "max": 1100,
            "currency": "USD",
            "period": "monthly"
        }
    }
    result = validate_json(valid_dict_correct_sample, resume_schema)
    logger.success(json.dumps(make_serializable(result), indent=2))


def get_resume_schema():
    return {
        "type": "object",
        "properties": {
            "scope_of_work": {"type": "array", "items": {"type": "string"}},
            "job_description": {"type": "string"},
            "qualifications": {"type": "array", "items": {"type": "string"}},
            "responsibilities": {"type": "array", "items": {"type": "string"}},
            "tech_stack": {
                "type": "object",
                "properties": {
                    "frontend": {"type": "array", "items": {"type": "string"}},
                    "backend": {"type": "array", "items": {"type": "string"}},
                    "database": {"type": "array", "items": {"type": "string"}},
                    "other": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["frontend", "backend", "database", "other"],
            },
            "level": {"type": "string", "enum": ["entry", "mid", "senior"]},
            "salary": {
                "type": "object",
                "properties": {
                    "min": {"type": "integer"},
                    "max": {"type": "integer"},
                    "currency": {"type": "string"},
                    "period": {"type": "string"},
                },
                "required": ["min", "max", "currency", "period"],
            },
        },
        "required": [
            "scope_of_work",
            "job_description",
            "qualifications",
            "responsibilities",
            "tech_stack",
            "level",
            "salary",
        ],
    }


if __name__ == "__main__":
    main_valid_dict_correct()
    main_valid_json_correct()
    main_valid_json_incorrect()
    main_invalid_json()
