# Example usage
import os
from jet.llm.mlx.prediction import predict_finishing_words, predict_next_word, predict_top_completions
from jet.logger import CustomLogger

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(
    script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)

if __name__ == "__main__":
    sentence = "The sun"

    # Predict next word
    next_word_result = predict_next_word(sentence, top_n=3)
    logger.log(f"Unfinished sentence:",
               next_word_result['input_sentence'], colors=["GRAY", "INFO"])
    logger.debug("Top next word predictions:")
    for pred in next_word_result['top_predictions']:
        logger.success(
            f"  Word: {pred['word']}, Probability: {pred['probability']}")
    logger.debug(f"Prompt: {next_word_result['prompt']}")
    logger.newline()

    # Predict finishing words
    finishing_result = predict_finishing_words(sentence, top_n=5)
    logger.log(f"Unfinished sentence:",
               finishing_result['input_sentence'], colors=["GRAY", "INFO"])
    logger.debug("Top finishing word predictions:")
    for pred in finishing_result['top_finishing_words']:
        logger.success(
            f"  Word: {pred['word']}, Probability: {pred['probability']}")
    logger.debug(f"Prompt: {finishing_result['prompt']}")
    logger.debug(f"Sample completion: {finishing_result['sample_completion']}")

    # Predict top completions
    completion_result = predict_top_completions(sentence, top_n=5)
    logger.log(f"Unfinished sentence:",
               completion_result['input_sentence'], colors=["GRAY", "INFO"])
    logger.debug("Top sentence completions:")
    for pred in completion_result['top_completions']:
        logger.success(
            f"  Completion: {pred['completion']}, Probability: {pred['probability']}")
    logger.debug(f"Prompt: {completion_result['prompt']}")
    logger.debug(
        f"Unique completions found: {completion_result['unique_completions_found']}")
