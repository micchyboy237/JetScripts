import json
from typing import Dict, Optional
from mlx_lm import load, generate

model, tokenizer = load("mlx-community/Llama-3.2-1B-Instruct-4bit")
max_tokens = -1

# Define a system and user prompt for generating tags
system_prompt = """
You are an expert tag extractor for web-scraped content. Analyze the text to generate a concise list of tags representing genres, themes, and demographics for text preprocessing in similarity search or indexing. Adapt to various content types (e.g., movies, articles, products, posts).

**Guidelines**:

1. **Tag Extraction**:
   - Extract 5-10 tags representing genres (e.g., Action, Adventure), themes (e.g., Isekai), and demographics (e.g., Shounen).
   - Focus on significant, specific terms; skip generic stopwords unless part of a phrase.
   - Include only relevant and confident tags.

2. **Output**:
   - Return only a JSON object with:
     - `tags`: Array of tag strings.
   - Output valid JSON only, no extra text.

3. **Flexibility**:
   - Handle structured, narrative, or incomplete content.
   - Skip uncertain tags.
""".strip()

user_prompt = """
## Naruto: Shippuuden Movie 6 - Road to Ninja
Movie, 2012 Finished 1 ep, 109 min
Action Adventure Fantasy
Naruto: Shippuuden Movie 6 - Road to Ninja
Returning home to Konohagakure, the young ninja celebrate defeating a group of supposed Akatsuki members. Naruto Uzumaki and Sakura Haruno, however, feel differently. Naruto is jealous of his comrades' congratulatory families, wishing for the presence of his own parents. Sakura, on the other hand, is angry at her embarrassing parents, and wishes for no parents at all. The two clash over their opposing ideals, but are faced with a more pressing matter when the masked Madara Uchiha suddenly appears and transports them to an alternate world. In this world, Sakura's parents are considered heroes--for they gave their lives to protect Konohagakure from the Nine-Tailed Fox attack 10 years ago. Consequently, Naruto's parents, Minato Namikaze and Kushina Uzumaki, are alive and well. Unable to return home or find the masked Madara, Naruto and Sakura stay in this new world and enjoy the changes they have always longed for. All seems well for the two ninja, until an unexpected threat emerges that pushes Naruto and Sakura to not only fight for the Konohagakure of the alternate world, but also to find a way back to their own. [Written by MAL Rewrite]
Studio Pierrot
Source Manga
Theme Isekai
Demographic Shounen
7.68
366K
Add to My List
""".strip()

few_shot_examples = [
    {
        "role": "user",
        "content": """
Input:
```
## Naruto: Shippuuden Movie 6 - Road to Ninja
Movie, 2012, 109 min
Action Adventure Fantasy
Returning home to Konohagakure, Naruto Uzumaki and Sakura Haruno...
Studio Pierrot
Source Manga
Theme Isekai
Demographic Shounen
7.68
```
""".strip()
    },
    {
        "role": "assistant",
        "content": """
```json
{
  "tags": ["Action", "Adventure", "Fantasy", "Isekai", "Shounen"]
}
```""".strip()
    },
    {
        "role": "user",
        "content": """
Input:
```
# The Rise of Quantum Computing
Published: 2023
By Dr. Alice Thompson
Quantum computing will revolutionize technology. IBM and Google lead...
Categories: Technology, Science
```
""".strip()
    },
    {
        "role": "assistant",
        "content": """
```json
{
  "tags": ["Technology", "Science", "Quantum Computing"]
}
```""".strip()
    }
]


def parse_response(response: str) -> Optional[Dict]:
    """
    Parses the model response to extract tags as a dictionary.

    Args:
        response: Raw response string from the model.

    Returns:
        Dictionary containing tags, or None if parsing fails.
    """
    try:
        # Attempt to find JSON content in the response
        json_start = response.find('{')
        json_end = response.rfind('}') + 1
        if json_start == -1 or json_end == -1:
            print("Error: No JSON content found in response.")
            return None

        json_str = response[json_start:json_end]
        parsed = json.loads(json_str)

        # Validate required key
        if 'tags' not in parsed:
            print("Error: Response missing required key 'tags'.")
            return None

        return parsed
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse JSON response: {e}")
        return None
    except Exception as e:
        print(f"Error: Unexpected error during parsing: {e}")
        return None


prompt_template = """
Input:
```
{user_prompt}
```
""".strip()

if tokenizer.chat_template is not None:
    user_input = prompt_template.format(user_prompt=user_prompt)

    messages = [
        {"role": "system", "content": system_prompt},
        *few_shot_examples,
        {"role": "user", "content": user_input}
    ]
    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True
    )

    # Generate response
    response = generate(model, tokenizer, prompt=prompt,
                        max_tokens=max_tokens, verbose=True)

    # Parse the response
    parsed_response = parse_response(response)

    if parsed_response:
        print("\nParsed Tags:")
        print(json.dumps(parsed_response, indent=2))
    else:
        print("Failed to parse response. Raw response:")
        print(response)
