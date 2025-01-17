from typing import Any, Dict
import inspect

from jet.logger import logger
from jet.transformers.formatters import format_json

# Assuming last_matching_frame is a frame from the inspect stack
last_matching_frame = inspect.stack()[-1]


def get_non_empty_attributes(obj: Any) -> Dict[str, Any]:
    """
    Extracts the non-empty attributes of an object (excluding those starting with "_")
    and returns them in a dictionary, filtering out attributes with values that are
    considered empty or falsy.

    Args:
        obj: The object from which to extract attributes.

    Returns:
        A dictionary with attribute names as keys and their corresponding
        non-falsy values as values.
    """
    # Filter out attributes that are falsy (None, False, 0, "", [], {}, set()) and those starting with "_"
    return {
        attr: getattr(obj, attr)
        for attr in dir(obj)
        if not attr.startswith('_') and getattr(obj, attr) not in [None, False, 0, "", [], {}, set()]
    }


def get_internal_attributes(obj: Any) -> Dict[str, Any]:
    """
    Extracts the attributes of an object that start with "_" and returns them in a dictionary.

    Args:
        obj: The object from which to extract attributes.

    Returns:
        A dictionary with attribute names starting with "_" as keys and their corresponding values.
    """
    return {
        attr: getattr(obj, attr)
        for attr in dir(obj)
        if attr.startswith('_')
    }


# Usage example:
if __name__ == "__main__":
    # Assuming `last_matching_frame` is defined somewhere

    # Get non-empty attributes (excluding internal ones)
    attributes_with_values = get_non_empty_attributes(last_matching_frame)
    logger.newline()
    logger.debug("Attributes with values:")
    logger.success(format_json(attributes_with_values))

    # Get internal attributes (those starting with "_")
    internal_attributes = get_internal_attributes(last_matching_frame)
    logger.newline()
    logger.debug("Internal attributes:")
    logger.success(format_json(internal_attributes))
