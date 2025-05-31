# Main function to run the stream chat


from jet.llm.mlx.helpers.generate_sliding_response import generate_sliding_response
from jet.llm.mlx.mlx_types import LLMModelType

# Sample system instruction and markdown context
SYSTEM_INSTRUCTION = """
You are an expert on anime. Below is a markdown document with anime titles under ## headers, each followed by details like release date, episodes, and synopsis. Use this context to answer queries accurately, preserving relationships between titles and their details. If the response is cut off due to token limits, end with '[CONTINUE]' and ensure the next part continues seamlessly.
"""

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


def main():
    model: LLMModelType = "qwen3-1.7b-4bit"

    # Initialize conversation history
    query = "Provide a detailed comparison of the anime titles in the provided markdown, focusing on their release dates, episode counts, and themes."
    messages = [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user", "content": f"{MARKDOWN_CONTEXT}\n\n{query}"}
    ]

    # Parameters
    max_tokens_per_generation = 200
    context_window = 300

    # Generate response
    response = generate_sliding_response(
        messages, max_tokens_per_generation, context_window, model)
    print("\n\nFull Response:\n", response)
    return response


if __name__ == "__main__":
    main()
