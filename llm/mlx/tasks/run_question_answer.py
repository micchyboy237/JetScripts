from jet.llm.mlx.tasks.question_answer import question_answer
from jet.logger import logger
from jet.transformers.formatters import format_json
from jet.llm.mlx.mlx_types import LLMModelType

if __name__ == "__main__":
    model: LLMModelType = "llama-3.2-3b-instruct-4bit"
    question: str = "What is the capital of France?"
    for method in ["stream_generate", "generate_step"]:
        logger.log("Method:", method, colors=["GRAY", "WHITE"])
        result = question_answer(
            question=question,
            model_path=model,
            method=method,
            max_tokens=20
        )
        logger.log("Question:", question, colors=["GRAY", "DEBUG"])
        logger.log("Answer:", result["answer"], colors=["GRAY", "SUCCESS"])
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
