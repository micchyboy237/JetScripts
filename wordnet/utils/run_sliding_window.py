from jet.wordnet.utils import sliding_window


def main_slide_on_long_context():
    MARKDOWN_CONTEXT = """
## Attack on Titan
- **Release Date**: April 7, 2013
- **Episodes**: 75
- **Synopsis**: Humanity fights against giant Titans...

## Demon Slayer
- **Release Date**: April 6, 2019
- **Episodes**: 44
- **Synopsis**: Tanjiro battles demons...

## Jujutsu Kaisen
- **Release Date**: October 3, 2020
- **Episodes**: 24
- **Synopsis**: Yuji Itadori consumes a cursed finger...

## My Hero Academia
- **Release Date**: April 3, 2016
- **Episodes**: 113
- **Synopsis**: Izuku Midoriya, a quirkless boy, inherits powers from the world's greatest hero...

## One Punch Man
- **Release Date**: October 5, 2015
- **Episodes**: 24
- **Synopsis**: Saitama, a hero who can defeat any opponent with one punch, seeks a worthy challenge...

## Fullmetal Alchemist: Brotherhood
- **Release Date**: April 5, 2009
- **Episodes**: 64
- **Synopsis**: Brothers Edward and Alphonse Elric use alchemy to restore their bodies after a failed experiment...
"""
    for sequence in sliding_window(MARKDOWN_CONTEXT, 300, 200):
        print(sequence)


def main_slide_on_short_context():
    text_corpus = "The quick brown fox jumps over the lazy dog. This is a simple text example for illustration."
    window_size = 3  # Number of tokens in each window
    step_size = 1    # Move the window by one token each time

    # Generate and print the sequences
    result = list(sliding_window(text_corpus, window_size, step_size))
    for sequence in sliding_window(text_corpus, window_size, step_size):
        print(sequence)


if __name__ == "__main__":
    main_slide_on_short_context()
    main_slide_on_long_context()
