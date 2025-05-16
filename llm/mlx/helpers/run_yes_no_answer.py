from jet.llm.mlx.helpers.yes_no_answer import AnswerResult, answer_yes_no
from jet.llm.mlx.mlx_types import ModelType
from jet.logger import logger

if __name__ == "__main__":
    model: ModelType = "llama-3.2-3b-instruct-4bit"
    question: str = "Is 1 + 1 equal to 2?"
    method = "generate_step"
    logger.log("Method:", method, colors=["GRAY", "WHITE"])
    result: AnswerResult = answer_yes_no(
        question, model, method=method)
    logger.log("Question:", question, colors=["GRAY", "DEBUG"])
    logger.log("Answer:", result["answer"], colors=["GRAY", "SUCCESS"])
    logger.log("Token ID:", result['token_id'], colors=["GRAY", "DEBUG"])
    logger.log("Valid:", result['is_valid'], colors=[
               "GRAY", "SUCCESS" if result['is_valid'] else "ERROR"])
    if result["error"]:
        print(f"Error: {result['error']}")
        logger.log("Error:", result['error'], colors=["GRAY", "ERROR"])
    else:
        logger.log("Output is valid.", method, colors=["GRAY", "SUCCESS"])
    logger.newline()
