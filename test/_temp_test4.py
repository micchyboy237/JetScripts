from jet.features.nltk_utils import get_word_sentence_combination_counts


if __name__ == "__main__":
    sample_text_str = "The quick brown fox jumps. The brown fox jumps quickly."
    sample_text_list = [
        "The quick brown fox jumps.",
        "The brown fox jumps quickly."
    ]
    print("Single string, non-sequential, n=2 (min_count=2):")
    bigram_counts_str_nonseq = get_word_sentence_combination_counts(
        sample_text_str, n=2, min_count=2, in_sequence=False)
    print(bigram_counts_str_nonseq)
    print("\nSingle string, sequential, n=2 (min_count=2):")
    bigram_counts_str_seq = get_word_sentence_combination_counts(
        sample_text_str, n=2, min_count=2, in_sequence=True)
    print(bigram_counts_str_seq)
    print("\nSingle string, non-sequential, n=None (min_count=2):")
    counts_str_nonseq_all = get_word_sentence_combination_counts(
        sample_text_str, n=None, min_count=2, in_sequence=False)
    print(counts_str_nonseq_all)
    print("\nList of strings, non-sequential, n=None (min_count=1):")
    counts_list_nonseq_all = get_word_sentence_combination_counts(
        sample_text_list, n=None, min_count=1, in_sequence=False)
    print(counts_list_nonseq_all)
    print("\nList of strings, sequential, n=None (min_count=1):")
    counts_list_seq_all = get_word_sentence_combination_counts(
        sample_text_list, n=None, min_count=1, in_sequence=True)
    print(counts_list_seq_all)
