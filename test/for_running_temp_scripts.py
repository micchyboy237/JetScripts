import json
from typing import List
from jet.llm.evaluators.context_relevancy_evaluator import evaluate_context_relevancy
from llama_index.core.prompts.base import PromptTemplate
from pydantic import BaseModel
from jet.llm.ollama.base import Ollama
from jet.logger import logger
from jet.transformers.formatters import format_json
from jet.utils.commands import copy_to_clipboard
from jet.utils.markdown import extract_json_block_content
from jet.validation.main.json_validation import validate_json


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
    "##"
]

PROMPT_TEMPLATE = PromptTemplate("""
Instruction:
- From the provided context, select the appropriate headers that are the most probable to contain contents that can answer the query in JSON format.
- By default, **the output must follow the ascending order of the rankings** as provided in the context, starting from "### 1" and going upwards.
- If the query explicitly requests the results to be in **descending order**, reverse the order accordingly.
- The output must maintain the exact order of the rankings as requested, and must include **verbatim strings** from the context.

=== OUTPUT FORMAT START ===
```json
{{
  "data": [
    "### 1. Sample 1",
    "### 2. Sample 2"
  ]
}}
```
=== OUTPUT FORMAT END ===

Example 1:
Context:
### 4. Toradora!
### 3. Teasing Master Takagi-san
### 2. Ikkoku House
### 1. Kaguya-sama: Love is War
Query: Get top 4 anime
Answer:
{{
  "data": [
    "### 1. Kaguya-sama: Love is War",
    "### 2. Ikkoku House",
    "### 3. Teasing Master Takagi-san",
    "### 4. Toradora!"
  ]
}}

Context:
{context}
Query: {query}
Answer:
""".strip())


class QueryResponse(BaseModel):
    data: List[str]


# Example Usage:
response = QueryResponse(data=["### 1. Sample 1", "### 2. Sample 2"])
print(response.json())


if __name__ == "__main__":
    output_cls = QueryResponse
    prompt_template = PROMPT_TEMPLATE

    llm_model = "gemma3:4b"
    llm = Ollama(temperature=0.3, model=llm_model)

    query = "What are the top 10 rom com anime today?"

    # Search doc headers
    contexts = header_texts
    eval_result = evaluate_context_relevancy(llm_model, query, contexts)
    if not eval_result.passing:
        query = ""
        response = llm.chat()
    context = json.dumps(contexts, indent=2)

    result = llm.structured_predict(
        output_cls,
        prompt=prompt_template,
        query=query,
        llm_kwargs={
            "context": context,
            "template": prompt_template,
            "template_vars": {"context": context, "query": query},
            "options": {
                "temperature": 0,
            }
        },
    )
    logger.success(format_json(result))
