from jet.logger import logger
from jet.transformers.formatters import format_json
from jet.validation.json_schema_validator import schema_validate_json


data = {
    "browser_query": "watch anime: otome, 2025, new releases, trending",
    "anime_list": [
        {
            "title": "Violet Evergarden",
            "release_year": 2016,
            "genre": ["Drama", "Romance", "Music"],
            "rating": 4.5
        },
        {
            "title": "Mushoku Tensei: Jobless Hero",
            "release_year": 2023,
            "genre": ["Fantasy", "Action", "Romance"],
            "rating": 4.2
        },
        {
            "title": "A Silent Voice",
            "release_year": 2016,
            "genre": ["Drama", "Romance", "Music"],
            "rating": 4.6
        }
    ],
    "required": [
        "anime_list"
    ]
}

schema = {
    "type": "object",
    "properties": {
        "anime_list": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "The title of the anime."
                    },
                    "release_year": {
                        "type": "integer",
                        "description": "The release year of the anime."
                    },
                    "genre": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "Genres associated with the anime."
                    },
                    "rating": {
                        "type": "number",
                        "description": "Average user rating of the anime."
                    }
                }
            }
        }
    },
    "required": [
        "anime_list"
    ]
}

validation_result = schema_validate_json(data, schema)

logger.success(format_json(validation_result))

assert validation_result["is_valid"] == True, f"Errors:\n{'\n'.join(validation_result['errors'])}"

logger.info("DONE!")
