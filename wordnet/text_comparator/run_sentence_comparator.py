from jet.wordnet.comparators.sentence_comparator import display_sentence_comparison

if __name__ == "__main__":
    # Example usage with equal-length lists
    base_sentences = [
        "The quick brown fox jumps.",
        "The sun rises slowly.",
        "A bird flies high."
    ]
    sentences_to_compare = [
        "The fast brown dog leaps.",
        "A dog barks loudly."
    ]
    print("Comparing equal-length sentence lists:")
    display_sentence_comparison(base_sentences, sentences_to_compare)

    # Example usage with unequal-length lists
    base_sentences = [
        "The quick brown fox jumps.",
        "The sun rises slowly."
    ]
    sentences_to_compare = [
        "The fast brown dog leaps."
    ]
    print("\nTrying to compare unequal-length sentence lists:")
    display_sentence_comparison(base_sentences, sentences_to_compare)

    # Example usage with unequal-length lists
    base_sentences = [
        "The quick brown fox jumps.",
    ]
    sentences_to_compare = [
        "The fast brown dog leaps.",
        "The sun rises slowly."
    ]
    print("\nTrying to compare unequal-length sentence lists:")
    display_sentence_comparison(base_sentences, sentences_to_compare)
