from jet.logger import logger
from jet.transformers.formatters import format_json
from jet.wordnet.sentence import is_last_word_in_sentence, split_sentences


def samples_with_abbreviations_and_acronyms():
    text = "Dr. Smith is from the U.S. He works at Acme Inc. He's great."
    sentences = split_sentences(text)

    logger.gray(f"Sentences ({len(sentences)}):")
    logger.success(format_json(sentences))

    print(f"Dr -> {is_last_word_in_sentence("Dr", text)}")
    print(f"Dr. -> {is_last_word_in_sentence("Dr.", text)}")
    print(f"U.S -> {is_last_word_in_sentence("U.S", text)}")
    print(f"U.S. -> {is_last_word_in_sentence("U.S.", text)}")
    print(f"Inc -> {is_last_word_in_sentence("Inc", text)}")
    print(f"Inc. -> {is_last_word_in_sentence("Inc.", text)}")
    print(f"great -> {is_last_word_in_sentence("great", text)}")
    print(f"great. -> {is_last_word_in_sentence("great.", text)}")


def samples_with_ordered_list_markers():
    text = "1. First item 1. 2. Second item. 3. Third item. a) Sub item. b) Another item b."
    sentences = split_sentences(text)

    logger.gray(f"Ordered List Sentences ({len(sentences)}):")
    logger.success(format_json(sentences))

    print(f"First -> {is_last_word_in_sentence('First', text)}")
    print(f"item. -> {is_last_word_in_sentence('item.', text)}")
    print(f"1. -> {is_last_word_in_sentence('1.', text)}")
    print(f"2. -> {is_last_word_in_sentence('2.', text)}")
    print(f"b. -> {is_last_word_in_sentence('b.', text)}")


def samples_with_unordered_list_markers():
    text = "- Bullet one. * Bullet two. + Bullet three."
    sentences = split_sentences(text)

    logger.gray(f"Unordered List Sentences ({len(sentences)}):")
    logger.success(format_json(sentences))

    print(f"Bullet -> {is_last_word_in_sentence('Bullet', text)}")
    print(f"three. -> {is_last_word_in_sentence('three.', text)}")


if __name__ == "__main__":

    print("\nRunning samples_with_abbreviations_and_acronyms...")
    samples_with_abbreviations_and_acronyms()

    print("\nRunning samples_with_ordered_list_markers...")
    samples_with_ordered_list_markers()

    print("\nRunning samples_with_unordered_list_markers...")
    samples_with_unordered_list_markers()
