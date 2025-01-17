from typing import Any, Dict
import inspect

from jet.logger import logger
from jet.transformers.formatters import format_json
from jet.utils.class_utils import get_internal_attributes, get_non_empty_attributes

# Assuming last_matching_frame is a frame from the inspect stack
last_matching_frame = inspect.stack()[-1]


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
