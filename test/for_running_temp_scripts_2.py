import json

from jet.llm.ollama.base import Ollama
from jet.logger import logger
from jet.transformers.formatters import format_json
from jet.utils.markdown import extract_json_block_content

header_texts = [
    "# Top 30 Best Rom-Com Anime Of All Time: The Ultimate Ranking",
    "### 30. Oh! My Goddess",
    "### 29. The Tatami Galaxy",
    "### 28. My Senpai is Annoying",
    "### 27. Net-juu no Susume",
    "### 26. His and Her Circumstances",
    "### 25. Arakawa Under the Bridge",
    "### 24. Maid Sama!",
    "### 23. Kamisama Kiss",
    "### 22. High Score Girl",
    "### 21. Kimi ni Todoke: From Me to You",
    "### 20. My Little Monster",
    "### 19. The Pet Girl of Sakurasou",
    "### 18. Monthly Girls' Nozaki-kun",
    "### 17. Ouran High School Host Club",
    "### 16. The Quintessential Quintuplets",
    "### 15. Tonikawa: Over the Moon For You",
    "### 14. Lovely Complex",
    "### 13. Working!!",
    "### 12. Tsurezure Children",
    "### 11. School Rumble",
    "### 10. Nisekoi: False Love",
    "### 9. Saekano: How to Raise a Boring Girlfriend",
    "### 8. Wotakoi: Love is Hard for Otaku",
    "### 7. My Love Story!!",
    "### 6. Horimiya",
    "### 5. My Teen Romantic Comedy SNAFU",
    "### 4. Toradora!",
    "### 3. Teasing Master Takagi-san",
    "### 2. Ikkoku House",
    "### 1. Kaguya-sama: Love is War",
    "### R. Romero",
    "### Keep Browsing",
    "### Related Posts",
    "#### Browse Fandoms",
]

prompt_template = """
Headers:
{headers}

Instruction:
{instruction}
Query: {query}
Answer:
""".strip()

if __name__ == "__main__":
    llm_model = "gemma3:4b"
    llm = Ollama(temperature=0.0, model=llm_model)

    headers = json.dumps(header_texts, indent=2)
    instruction = "Given the provided headers, select ones that are relevant to the query in JSON format."
    query = "What are the top 10 rom com anime today?"

    message = prompt_template.format(
        headers=headers,
        query=query,
        instruction=instruction,
    )

    response = llm.chat(message, model=llm_model)
    json_result = extract_json_block_content(str(response))
    result = json.loads(json_result)

    logger.success(format_json(result))
