from jet.llm.mlx.tasks.conditional_text_generation import conditional_text_generation
from jet.logger import logger
from jet.transformers.formatters import format_json
from jet.llm.mlx.mlx_types import LLMModelType
from typing import Dict

if __name__ == "__main__":
    model: LLMModelType = "llama-3.2-3b-instruct-4bit"
    prompt: str = "Describe a futuristic city."
    conditions: Dict[str, str] = {
        "Style": "Vivid and descriptive",
        "Tone": "Optimistic",
        "Keywords": "hovercars, neon lights, AI assistants"
    }
    for method in ["stream_generate", "generate_step"]:
        logger.log("Method:", method, colors=["GRAY", "WHITE"])
        result = conditional_text_generation(
            prompt=prompt,
            conditions=conditions,
            model_path=model,
            method=method,
            max_tokens=100
        )
        logger.log("Prompt:", prompt, colors=["GRAY", "DEBUG"])
        logger.log("Conditions:", conditions, colors=["GRAY", "DEBUG"])
        logger.log("Output:", result["output"], colors=["GRAY", "SUCCESS"])
        logger.log("Token IDs:", result['token_ids'], colors=["GRAY", "DEBUG"])
        logger.log("Valid:", result['is_valid'], colors=[
                   "GRAY", "SUCCESS" if result['is_valid'] else "ERROR"])
        if result["error"]:
            logger.log("Error:", result['error'], colors=["GRAY", "ERROR"])
        else:
            logger.log("Output is valid.", colors=["GRAY", "SUCCESS"])
        logger.newline()
        logger.gray("Result:")
        logger.success(format_json(result))
