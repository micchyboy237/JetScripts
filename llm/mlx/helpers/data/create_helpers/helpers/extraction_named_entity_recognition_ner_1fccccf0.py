import json
import re
from jet.llm.mlx import MlxNamedEntityRecognition
from jet.llm.mlx import MlxNamedEntityRecognitionConfig
from jet.llm.mlx import MlxNamedEntityRecognitionOutput
from jet.llm.mlx import MlxNamedEntityRecognitionError
from jet.llm.mlx import MlxNamedEntityRecognitionResult
from typing import Dict, List

def named_entity_recognition(input_text: str) -> Dict[str, List[str]]:
    """
    Performs Named Entity Recognition (NER) on the given input text.

    Args:
    input_text (str): The input text to be recognized.

    Returns:
    Dict[str, List[str]]: A dictionary containing the recognized entities and their corresponding labels.
    """

    # Initialize the Mlx Named Entity Recognition object
    mlx_named_entity_recognition = MlxNamedEntityRecognition()

    # Configure the Mlx Named Entity Recognition object
    mlx_named_entity_recognition_config = MlxNamedEntityRecognitionConfig()
    mlx_named_entity_recognition_config.entity_type = "EXTRACT"
    mlx_named_entity_recognition_config.system_message = "Identify and extract entities (e.g., Person, Product, Location) from the text, returning them in a structured format."
    mlx_named_entity_recognition_config.output_format = "json"
    mlx_named_entity_recognition_config.output_path = "output.json"

    try:
        # Perform the Named Entity Recognition
        mlx_named_entity_recognition_result = mlx_named_entity_recognition(input_text, mlx_named_entity_recognition_config)

        # Check for errors
        if not mlx_named_entity_recognition_result:
            raise MlxNamedEntityRecognitionError(mlx_named_entity_recognition_result)

        # Extract the recognized entities and their corresponding labels
        entities = {}
        for entity in mlx_named_entity_recognition_result:
            if entity['label'] == "PERSON":
                entities['entities'] = [entity['value']]
            elif entity['label'] == "PRODUCT":
                entities['entities'] = [entity['value']]
            elif entity['label'] == "LOCATION":
                entities['entities'] = [entity['value']]
            else:
                entities['entities'] = [entity['value']]

        # Return the recognized entities and their corresponding labels
        return {
            "entities": entities,
            "output_path": "output.json"
        }

    except MlxNamedEntityRecognitionError as e:
        # Handle errors with try-except blocks
        print(f"Error: {e}")
        return None

    except Exception as e:
        # Handle other errors with a generic error message
        print(f"An error occurred: {e}")
        return None


# Example usage
if __name__ == "__main__":
    # Input text
    input_text = "Elon Musk launched a Tesla Cybertruck in New York."

    # Perform Named Entity Recognition
    result = named_entity_recognition(input_text)

    # Check if the result is valid
    if result:
        # Print the recognized entities and their corresponding labels
        print(json.dumps(result, indent=4)))
    else:
        # Print an error message if no recognized entities are found
        print("No recognized entities found.")
```

This Python script implements the Named Entity Recognition (NER) structure as specified. It performs the NER on the given input text and returns the recognized entities and their corresponding labels in a structured format. The script also handles errors with try-except blocks and provides a generic error message if no recognized entities are found. The example usage demonstrates how to use the script to recognize entities and their corresponding labels.