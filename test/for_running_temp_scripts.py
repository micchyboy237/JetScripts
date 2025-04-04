import re
from typing import Optional, Tuple

from jet.logger import logger
from jet.transformers.formatters import format_json


def _default_parser_function(output_str: str) -> Tuple[Optional[float], Optional[str]]:
    # Pattern to match the feedback and response
    # This pattern looks for any text ending with '[RESULT]' followed by a number
    pattern = r"([\s\S]+)(?:\[RESULT\]\s*)([\d.]+)"

    # Using regex to find all matches
    result = re.search(pattern, output_str)

    # Check if any match is found
    if result:
        # Assuming there's only one match in the text, extract feedback and response
        feedback, score = result.groups()
        score = float(score) if score is not None else score
        return score, feedback.strip()
    else:
        return None, None


# Test input text
input_text = """
Feedback:
The retrieved context is highly relevant to the query "What are the steps in registering a National ID in the Philippines?". The document provides a detailed, step-by-step guide on the process.

1.  **Does the retrieved context match the subject matter of the user's query?** (2/2) - Absolutely. The entire document focuses on the process of registering for a Philippine National ID (PhilID), directly addressing the user's question.

2.  **Can the retrieved context be used exclusively to provide a full answer to the user’s query?** (2/2) - Yes. The document outlines all the necessary steps, from preparing documents to receiving the ID card, effectively providing a complete answer.

[RESULT] 2.00
"""

# Apply the function
result = _default_parser_function(input_text)

logger.newline()
logger.success(format_json(result))
