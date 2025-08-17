from jet.llm.mlx.tasks.generative_tasks import generative_tasks
from jet.logger import logger
from jet.transformers.formatters import format_json
from jet.models.model_types import LLMModelType

if __name__ == "__main__":
    model: LLMModelType = "llama-3.2-3b-instruct-4bit"
    prompt: str = "Write a short story about a robot learning to paint in a futuristic city."
    for method in ["stream_generate", "generate_step"]:
        logger.log("Method:", method, colors=["GRAY", "WHITE"])
        result = generative_tasks(
            prompt=prompt,
            model_path=model,
            method=method,
            max_tokens=150
        )
        logger.log("Prompt:", prompt, colors=["GRAY", "DEBUG"])
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
