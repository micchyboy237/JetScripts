from jet.llm.mlx.helpers.text_segmentation import SegmentationResult, text_segmentation
from jet.llm.mlx.mlx_types import ModelType
from jet.logger import logger

if __name__ == "__main__":
    model: ModelType = "llama-3.2-3b-instruct-4bit"
    input_text: str = (
        "The sun sets slowly behind the mountain, casting a warm golden glow over the valley. "
        "Birds chirp softly as the cool evening breeze begins to blow. "
        "People in the village below prepare for a quiet night, gathering around fires."
    )

    for method in ["stream_generate", "generate_step"]:
        logger.log("Method:", method, colors=["GRAY", "WHITE"])
        result: SegmentationResult = text_segmentation(
            input_text, model, method=method
        )
        logger.log("Input Text:", input_text, colors=["GRAY", "DEBUG"])
        logger.log("Segments:", colors=["GRAY", "SUCCESS"])
        for i, segment in enumerate(result["segments"], 1):
            logger.log(f"{i}.", segment, colors=["GRAY", "SUCCESS"])
        logger.log("Valid:", result["is_valid"], colors=[
                   "GRAY", "SUCCESS" if result["is_valid"] else "ERROR"])
        if result["error"]:
            logger.log("Error:", result["error"], colors=["GRAY", "ERROR"])
        else:
            logger.log("Output is valid.", method, colors=["GRAY", "SUCCESS"])
        logger.newline()