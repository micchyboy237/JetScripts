from jet.llm.mlx.tasks.text_summarization import SummarizationResult, text_summarization
from jet.models.model_types import LLMModelType
from jet.logger import logger

if __name__ == "__main__":
    model: LLMModelType = "llama-3.2-3b-instruct-4bit"
    input_text: str = (
        "The sun sets slowly behind the mountain, casting a warm golden glow over the valley. "
        "Birds chirp softly as the cool evening breeze begins to blow, signaling the end of the day. "
        "People in the village below prepare for a quiet night, gathering around fires and sharing stories."
    )

    for method in ["stream_generate", "generate_step"]:
        logger.log("Method:", method, colors=["GRAY", "WHITE"])
        result: SummarizationResult = text_summarization(
            input_text, model, method=method
        )
        logger.log("Input Text:", input_text, colors=["GRAY", "DEBUG"])
        logger.log("Summary:", result["summary"], colors=["GRAY", "SUCCESS"])
        logger.log("Valid:", result["is_valid"], colors=[
                   "GRAY", "SUCCESS" if result["is_valid"] else "ERROR"])
        if result["error"]:
            logger.log("Error:", result["error"], colors=["GRAY", "ERROR"])
        else:
            logger.log("Output is valid.", method, colors=["GRAY", "SUCCESS"])
        logger.newline()
