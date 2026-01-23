from jet.wordnet.utils import increasing_window


if __name__ == "__main__":
    text_corpus = "The quick brown fox jumps over the lazy dog. This is a simple text example for illustration."
    step_size = 1    # Move the window by one token each time

    # Generate and print the sequences
    result = list(increasing_window(text_corpus, step_size))
    for sequence in increasing_window(text_corpus, step_size=step_size):
        print(sequence)
