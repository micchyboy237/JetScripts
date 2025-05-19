import re
from nltk.tokenize import sent_tokenize
from words import count_words

# nltk.download('punkt')


def process_sentence_newlines(sentences):
    processed_sentences = []
    carry_over_text = ''

    for sentence in sentences:
        newline_index = sentence.find('NEWLINE_TOKEN')

        if newline_index != -1:
            # Add the sentence up to the NEWLINE_TOKEN with any carry-over text
            processed_sentences.append(
                (carry_over_text + ' ' + sentence[:newline_index]).strip())

            # Update carry_over_text with the text after NEWLINE_TOKEN
            carry_over_text = sentence[newline_index:].strip()
        else:
            # Add the sentence with any carry-over text
            processed_sentences.append(
                (carry_over_text + ' ' + sentence).strip())
            carry_over_text = ''  # Reset carry-over text

    # Add the last carry-over text if it's not empty
    if carry_over_text and carry_over_text.replace('NEWLINE_TOKEN', '').strip():
        processed_sentences.append(carry_over_text)

    return processed_sentences


def handle_long_sentence(sentence, count_tokens_func, max_tokens):
    words = sentence.split()
    sub_sentence = ''
    segment = ''

    for word in words:
        combined_text = sub_sentence + (' ' if sub_sentence else '') + word
        if count_tokens_func(combined_text) <= max_tokens:
            sub_sentence = combined_text
        else:
            if sub_sentence:
                segment += (' ' + sub_sentence) if segment else sub_sentence
                sub_sentence = word
            else:
                # If a single word exceeds max_tokens, consider it as a segment
                segment += (' ' + word) if segment else word
                sub_sentence = ''

    # Return the last sub-sentence if it's not empty
    return segment + (' ' + sub_sentence) if sub_sentence else segment


def ngrams(sentence, n):
    """Generate n-grams from a sentence."""
    words = sentence.split()
    return [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]


def get_list_marker_pattern_substring():
    roman_numeral = r'(X{0,3}(IX|IV|V?I{0,3}))'
    pattern = r'^(\d+|[a-zA-Z]|' + roman_numeral + r')[\.\)]+\s?'

    return pattern


def get_list_marker_pattern():
    pattern = get_list_marker_pattern_substring() + r'$'

    return pattern


def get_list_sentence_pattern():
    pattern = get_list_marker_pattern_substring() + r'[\w]+'

    return pattern


def is_ordered_list_marker(marker):
    pattern = get_list_marker_pattern()

    return bool(re.match(pattern, marker.strip(), re.IGNORECASE))


def is_ordered_list_sentence(sentence):
    pattern = get_list_sentence_pattern()

    return bool(re.match(pattern, sentence.strip(), re.IGNORECASE))


def adaptive_split(text, count_tokens_func=count_words, max_tokens=0):
    # Check for empty or whitespace-only strings
    if not text.strip():
        return []

    max_tokens = max_tokens or 0

    # Replace \n with NEWLINE_TOKEN
    text = text.replace('\n', 'NEWLINE_TOKEN')

    # Use NLTK for more robust sentence splitting
    raw_sentences = sent_tokenize(text)
    processed_sentences = process_sentence_newlines(raw_sentences)

    segments = []
    current_segment = ''

    for sentence in processed_sentences:
        # Skip empty sentences
        if not sentence.strip():
            continue

        # Calculate token counts
        current_tokens_len = count_tokens_func(current_segment)
        tokens_len = count_tokens_func(sentence)

        stripped_sentence = sentence.replace('NEWLINE_TOKEN', '')
        # Check if the sentence is a continuation of an ordered list item
        if is_ordered_list_marker(stripped_sentence):
            if is_ordered_list_sentence(stripped_sentence):
                # Append the current segment if it's not empty
                if current_segment:
                    segments.append(current_segment)

                    current_segment = ''

                segments.append(sentence)
            else:
                current_segment += (' ' if current_segment else '') + sentence
        # Check if adding the sentence exceeds the max token limit
        elif current_tokens_len + tokens_len <= max_tokens:
            current_segment += (' ' if current_segment else '') + sentence
        else:
            # Append the current segment if it's not empty
            if current_segment and not is_ordered_list_marker(current_segment):
                segments.append(current_segment)

                current_segment = ''
            # Handle the next sentence
            if tokens_len <= max_tokens:
                current_segment = sentence
            else:
                # Split long sentence into words and build segments
                long_segment = handle_long_sentence(
                    sentence, count_tokens_func, max_tokens)

                current_segment += (' ' if current_segment else '') + \
                    long_segment

    # Append the last segment if it exists
    if current_segment:
        segments.append(current_segment)

    # Replace NEWLINE_TOKEN with \n on all segments
    segments = [segment.replace('NEWLINE_TOKEN', '\n')
                for segment in segments]

    return segments


def split_sentences(text):
    sentences = sent_tokenize(text)
    adjusted_sentences = []

    i = 0
    while i < len(sentences):
        current_sentence = sentences[i]
        # Check if the current sentence is a list marker and not the last sentence
        if i + 1 < len(sentences) and is_ordered_list_sentence(current_sentence):
            # Merge with the next sentence
            adjusted_sentences.append(
                current_sentence + ' ' + sentences[i + 1])
            i += 2  # Skip the next sentence as it's merged
        else:
            adjusted_sentences.append(current_sentence)
            i += 1

    return adjusted_sentences


def count_sentences(text):
    sentences = adaptive_split(text)

    return len(sentences)


def get_sentences(text: str, n: int) -> list[str]:
    sentences = adaptive_split(text)

    return sentences[:n]


if __name__ == "__main__":
    text = 'Ang mga pang-uri o adjectives sa Ingles ay salitang nagbibigay turing o naglalarawan sa isang pangngalan o panghalip. Ito ay nagsasaad ng uri o katangian ng tao, bagay, hayop, pook, o pangyayari.'

    sentence_count = count_sentences(text)
    print(f"Number of sentences: {sentence_count}")

    sentences = get_sentences(text, 3)
    print(sentences)
