# Example usage
from jet.llm.mlx.prediction import predict_finishing_words, predict_next_word, predict_top_completions


if __name__ == "__main__":
    sentence = "The sun"

    # Predict next word
    next_word_result = predict_next_word(sentence, top_n=3)
    print(f"Unfinished sentence: {next_word_result['input_sentence']}")
    print("Top next word predictions:")
    for pred in next_word_result['top_predictions']:
        print(f"  Word: {pred['word']}, Probability: {pred['probability']}")
    print(f"Prompt: {next_word_result['prompt']}")
    print()

    # Predict finishing words
    finishing_result = predict_finishing_words(sentence, top_n=5)
    print(f"Unfinished sentence: {finishing_result['input_sentence']}")
    print("Top finishing word predictions:")
    for pred in finishing_result['top_finishing_words']:
        print(f"  Word: {pred['word']}, Probability: {pred['probability']}")
    print(f"Prompt: {finishing_result['prompt']}")
    print(f"Sample completion: {finishing_result['sample_completion']}")

    # Predict top completions
    completion_result = predict_top_completions(sentence, top_n=5)
    print(f"Unfinished sentence: {completion_result['input_sentence']}")
    print("Top sentence completions:")
    for pred in completion_result['top_completions']:
        print(
            f"  Completion: {pred['completion']}, Probability: {pred['probability']}")
    print(f"Prompt: {completion_result['prompt']}")
    print(
        f"Unique completions found: {completion_result['unique_completions_found']}")
