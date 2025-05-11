from jet.llm.mlx.helpers.dialogue import dialogue, ChatMessage
from jet.logger import logger
from jet.transformers.formatters import format_json
from jet.llm.mlx.mlx_types import ModelType
from typing import List

if __name__ == "__main__":
    model: ModelType = "llama-3.2-3b-instruct-4bit"
    history: List[ChatMessage] = [
        {"role": "user", "content": "Hi, I'm planning a trip to Paris."},
        {"role": "assistant", "content": "That sounds amazing! Paris is beautiful. When are you planning to go?"}]
    message: str = "I'm thinking about going in spring. Any recommendations?"
    for method in ["stream_generate", "generate_step"]:
        logger.log("Method:", method, colors=["GRAY", "WHITE"])
        result = dialogue(
            message=message,
            history=history,
            model_path=model,
            method=method,
            max_tokens=100
        )
        logger.log("Conversation History:", colors=["GRAY", "DEBUG"])
        for msg in history:
            logger.log(f"{msg['role'].capitalize()}: {msg['content']}", colors=["GRAY", "DEBUG"])
        logger.log("New Message:", message, colors=["GRAY", "DEBUG"])
        logger.log("Response:", result["response"], colors=["GRAY", "SUCCESS"])
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