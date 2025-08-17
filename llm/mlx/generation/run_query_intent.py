from typing import Literal, TypedDict
from jet.llm.mlx.base import MLX
from jet.llm.mlx.generation.query_intent import get_query_intent
from jet.llm.mlx.helpers.base import get_system_date_prompt
from jet.models.model_types import LLMModelType

if __name__ == "__main__":
    # Example 1: Informational Query
    query1 = "What is the capital of France?"
    result1 = get_query_intent(query1, model="qwen3-1.7b-4bit")
    print(result1)  # Expected: {"intent": "informational", "confidence": 0.95}

    # Example 2: Navigational Query
    query2 = "Facebook login"
    result2 = get_query_intent(query2)
    print(result2)  # Expected: {"intent": "navigational", "confidence": 0.90}

    # Example 3: Transactional Query
    query3 = "Buy iPhone 13"
    result3 = get_query_intent(query3, model="qwen3-1.7b-4bit")
    print(result3)  # Expected: {"intent": "transactional", "confidence": 0.92}

    # Example 4: Commercial Investigation Query
    query4 = "Best laptops 2023"
    result4 = get_query_intent(query4)
    # Expected: {"intent": "commercial_investigation", "confidence": 0.88}
    print(result4)

    # Example 5: Ambiguous Query (Fallback Case)
    query5 = "xyz"
    result5 = get_query_intent(query5)
    print(result5)  # Expected: {"intent": "informational", "confidence": 0.5}
