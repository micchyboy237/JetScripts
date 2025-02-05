from collections import defaultdict
from typing import List, Dict, Any
import json


def merge_dicts_recursive(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merges two dictionaries."""
    for key, value in dict2.items():
        if isinstance(value, dict) and key in dict1 and isinstance(dict1[key], dict):
            dict1[key] = merge_dicts_recursive(dict1[key], value)
        else:
            dict1[key] = value
    return dict1


def dict_to_tuple(d: Dict[str, Any]) -> tuple:
    """Convert dictionary to a sorted tuple of key-value pairs, recursively handle nested dictionaries."""
    return tuple((k, dict_to_tuple(v) if isinstance(v, dict) else v) for k, v in sorted(d.items()))


def merge_objects_by_attribute(array_to_merge: List[Dict[str, Any]], key_attr: str = "name") -> Dict[str, Any]:
    merged_data = defaultdict(lambda: defaultdict(list))

    for obj in array_to_merge:
        for path, values in obj.items():
            entity = None
            relation = None

            for value in values:
                if isinstance(value, dict) and key_attr in value:
                    key_value = value[key_attr]
                    if entity is None:
                        entity = key_value
                    merged_data[entity][path].append(value)
                elif isinstance(value, str):
                    relation = value

                if relation and entity:
                    merged_data[entity][path].append(relation)
                    relation = None

    # Recursively merge attributes within each entity's paths
    for entity, paths in merged_data.items():
        for path, values in paths.items():
            # Handle case where values are dictionaries and strings
            unique_values = []
            for value in values:
                if isinstance(value, dict):
                    unique_values.append(value)
                elif isinstance(value, str):
                    unique_values.append(value)
            merged_data[entity][path] = list(
                {dict_to_tuple(d): d for d in unique_values if isinstance(d, dict)}.values())

            # Now merge dictionaries recursively within each path
            if merged_data[entity][path]:
                merged_data[entity][path] = merge_dicts_recursive(
                    {}, merged_data[entity][path][0])

    return merged_data


# Example usage
array_to_merge = [
    {
        "path1": [
            {
                "first_name": "Jethro Reuel",
                "last_name": "Estrada",
                "middle_name": "Arao",
                "name": "Jethro Reuel A. Estrada",
                "preferred_name": "Jethro or Jet"
            },
            "WORKED_AT",
            {
                "end_date": {
                    "_Date__ordinal": 738977,
                    "_Date__year": 2024,
                    "_Date__month": 4,
                    "_Date__day": 1
                },
                "name": "Built Different LLC",
                "position": "Web / Mobile Developer",
                "start_date": {
                    "_Date__ordinal": 738611,
                    "_Date__year": 2023,
                    "_Date__month": 4,
                    "_Date__day": 1
                }
            },
            "OWNS",
            {
                "achievements": "Developed iOS app from scratch, delivered web app for demo",
                "name": "JABA AI"
            },
            "USES",
            {
                "name": "React"
            }
        ],
        "path2": [
            {
                "age": 34,
                "birthday": "1990-12-01",
                "country": "Philippines",
                "gender": "Male",
                "nationality": "Filipino",

            },
            "KNOWS",
            {
                "name": "React"
            }
        ]
    }
]

# array_to_merge = [
#     {"path1": [{"name": "entity1", "age": 30}], "path2": ["relation1"]},
#     {"path1": [{"name": "entity1", "age": 35}], "path2": ["relation2"]},
#     {"path1": [{"name": "entity2", "age": 25}], "path2": ["relation3"]},
# ]

if __name__ == "__main__":
    merged_result = merge_objects_by_attribute(array_to_merge)
    print(json.dumps(merged_result, indent=2))
