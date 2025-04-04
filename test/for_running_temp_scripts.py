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
Okay, let's evaluate the context against the query: "What are the steps in registering a National ID in the Philippines?".

**Question 1: Does the retrieved context match the subject matter of the user's query?**

The context is *highly* relevant. It directly addresses the query by outlining the steps involved in registering for a National ID (PhilID) in the Philippines. The document explicitly provides a numbered list of steps: "1. Prepare the documents required for PhilID, 2. Go to any PhilID Registration Center, 3. Fill out your Philippine National ID Application Form, 4. Have your biometrics information captured, 5. Generate a PSN (PhilSys Number), 6. Receive your National ID Card".

**Score: 2/2**

**Question 2: Can the retrieved context be used exclusively to provide a full answer to the user's query?**

Yes, the context provides a complete answer to the query. It details *all* the necessary steps for registering a National ID. It also includes relevant information about required documents and the process of generating a PSN (PhilSys Number).  There's no need for additional context to fully answer the question.

**Score: 2/2**

[RESULT] 4.00
"""

# Apply the function
result = _default_parser_function(input_text)

logger.newline()
logger.success(format_json(result))
