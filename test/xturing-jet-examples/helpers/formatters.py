from words import count_words
from unidecode import unidecode
import nltk
import random
from collections import Counter


def decode_encoded_characters(text):
    rules = [
        {'&#39;', '\''},
        {'&amp;', '&'},
        {'&lt;', '<'},
        {'&gt;', '>'},
        ('', '\''),
        ('，', ','),
        ('。', '.'),
        ('、', ','),
        ('”', '"'),
        ('“', '"'),
        ('∶', ':'),
        ('：', ':'),
        ('？', '?'),
        ('《', '"'),
        ('》', '"'),
        ('）', ')'),
        ('！', '!'),
        ('（', '('),
        ('；', ';'),
        ('１', '1'),
        ('」', '"'),
        ('「', '"'),
        ('０', '0'),
        ('３', '3'),
        ('２', '2'),
        ('５', '5'),
        ('６', '6'),
        ('９', '9'),
        ('７', '7'),
        ('８', '8'),
        ('４', '4'),
        ('．', '.'),
        ('～', '~'),
        ('’', '\''),
        ('‘', '\''),
        ('…', '...'),
        ('━', '-'),
        ('〈', '<'),
        ('〉', '>'),
        ('【', '['),
        ('】', ']'),
        ('％', '%')
    ]

    for (rule, replacement) in rules:
        text = text.replace(rule, replacement)

    return text


def clean_string(text: str):
    text = decode_encoded_characters(text)

    def remove_unmatched_characters(text, open_char, close_char):
        balance = 0
        new_text = ''
        for char in text:
            if char == open_char:
                balance += 1
                new_text += char
            elif char == close_char:
                if balance == 0:
                    continue  # Ignore this character as it's unmatched
                balance -= 1
                new_text += char
            else:
                new_text += char
        return new_text

    # Count the number of quotes
    quote_count = text.count('\"')

    # If the number of quotes is odd, remove the last one
    if quote_count % 2 != 0:
        last_quote_index = text.rfind('\"')
        text = text[:last_quote_index] + text[last_quote_index + 1:]

    # Process opening and closing characters
    for char_pair in [('(', ')'), ('[', ']'), ('{', '}')]:
        text = remove_unmatched_characters(text, *char_pair)
        text = remove_unmatched_characters(
            text[::-1], char_pair[1], char_pair[0])[::-1]

    # Remove leading/trailing commas
    text = text.strip(",")

    # Remove enclosing double quotes only if they're at both ends of the string
    if text.startswith("\"") and text.endswith("\""):
        text = text[1:-1]

    # Remove leading quote only if it's the only one in the string
    elif text.startswith("\"") and text.count('\"') == 1:
        text = text[1:]

    # Remove trailing quote only if it'text the only one in the string
    elif text.endswith("\"") and text.count('\"') == 1:
        text = text[:-1]

    # Remove enclosing single quotes only if they're at both ends of the string
    elif text.startswith("'") and text.endswith("'"):
        text = text[1:-1]

    # Remove leading single quote only if it's the only one in the string
    elif text.startswith("'") and text.count("'") == 1:
        text = text[1:]

    # Remove trailing single quote only if it'text the only one in the string
    elif text.endswith("'") and text.count("'") == 1:
        text = text[:-1]

    # Remove extra spaces while preserving newlines
    lines = text.split('\n')
    cleaned_lines = [' '.join(line.split()) for line in lines]
    text = '\n'.join(cleaned_lines)

    return text.strip()


def preserve_tagalog_chars(s: str) -> str:
    special_char_mapping = {
        "ñ": "<ntilde>",
        "Ñ": "<NTILDE>",

    }

    for char, placeholder in special_char_mapping.items():
        s = s.replace(char, placeholder)

    s = unidecode(s)

    for char, placeholder in special_char_mapping.items():
        s = s.replace(placeholder, char)

    return s


def clean_sample(text):
    text = text.strip()

    if not text:
        return ''

    text = clean_string(text)
    text = preserve_tagalog_chars(text)
    # text = lemmatize_text(text)

    return text


def extract_content(text: str, max_words=130):
    text = clean_sample(text)
    # Check if the input is a string
    if not isinstance(text, str):
        raise ValueError("Input must be a string")

    split_text = text.split('\n')
    preserved_sentences = []

    for part in split_text:
        for sentence in nltk.tokenize.sent_tokenize(part):
            preserved_sentences.append(sentence)
        preserved_sentences.append('\n')  # Preserve line breaks

    # Remove the last '\n' added
    if preserved_sentences:
        preserved_sentences.pop()

    # Initialize variables
    word_count = 0
    extracted_content = []
    add_newline = False

    # Process each sentence
    for sentence in preserved_sentences:
        # Handle newline separately
        if sentence == '\n':
            add_newline = True
            continue

        sentence_word_count = count_words(sentence)
        new_count = word_count + sentence_word_count

        # Add sentence to extracted content
        if add_newline:
            extracted_content.append('\n')
            add_newline = False
        extracted_content.append(sentence)
        word_count = new_count

        # Break if the word limit is exceeded after adding this sentence
        if new_count > max_words:
            break

    # Join the extracted content with spaces
    extracted_content_with_spaces = ''
    for sentence in extracted_content:
        if sentence == '\n':
            extracted_content_with_spaces += '\n'
        else:
            if extracted_content_with_spaces.endswith('\n'):
                extracted_content_with_spaces += sentence
            else:
                extracted_content_with_spaces += ' ' + sentence

    return extracted_content_with_spaces.strip()


def analyze_sentence_endings(data):
    endings = [sentence[-1] for sentence in data if sentence]
    counts = Counter(endings)
    return counts


def diversify_endings(data, desired_distribution={'?': 0.25, '!': 0.25, '.': 0.5}):
    current_distribution = Counter([sentence[-1] for sentence in data])
    total = sum(current_distribution.values())
    current_distribution = {k: v / total for k,
                            v in current_distribution.items()}
    for i, sentence in enumerate(data):
        if current_distribution[sentence[-1]] > desired_distribution[sentence[-1]]:
            choice = random.choice(list(desired_distribution.keys()))
            data[i] = sentence[:-1] + choice
    return data


def balance_languages(data, target_distribution={'en': 0.5, 'es': 0.5}):
    language_counts = Counter([item['language'] for item in data])
    total = sum(language_counts.values())
    current_distribution = {k: v / total for k, v in language_counts.items()}
    # Placeholder logic for testing
    return current_distribution


if __name__ == "__main__":
    text = "1. Eat a balanced and nutritious diet: Make sure your meals are inclusive of a variety of fruits and vegetables, lean protein, whole grains, and healthy fats. This helps to provide your body with the essential nutrients to function at its best and can help prevent chronic diseases.\n\n2. Engage in regular physical activity: Exercise is crucial for maintaining strong bones, muscles, and cardiovascular health. Aim for at least 150 minutes of moderate aerobic exercise or 75 minutes of vigorous exercise each week.\n\n3. Get enough sleep: Getting enough quality sleep is crucial for physical and mental well-being. It helps to regulate mood, improve cognitive function, and supports healthy growth and immune function. Aim for 7-9 hours of sleep each night."

    preprocessed_content = clean_sample(text)
    extracted_content = extract_content(text, max_words=50)

    print(f"Preprocessed content:\n{preprocessed_content}")
    print('\n')
    print(f"Extracted content:\n{extracted_content}")
