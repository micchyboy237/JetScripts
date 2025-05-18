from jet.llm.ollama.base import chat
from jet.logger import logger
from jet.transformers.formatters import format_json

SYSTEM = """
You are a helpful assistant that receives a browser-style query and generates a JSON Schema describing the kind of structured data users typically expect from websites related to that query.

Your output must:
- Use JSON Schema Draft 2020-12 or later.
- Include only the most relevant required and optional fields.
- Be concise and use appropriate types and descriptions.
- Respond with the JSON Schema only — no extra commentary.

---

Example Query 1:
"Philippines TikTok online seller registration steps 2025"

Example Response 1:
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "TikTok Seller Registration Steps Schema",
  "type": "object",
  "required": ["steps"],
  "properties": {
    "steps": {
      "type": "array",
      "description": "Ordered list of registration steps",
      "items": {
        "type": "object",
        "required": ["step_number", "instruction"],
        "properties": {
          "step_number": {
            "type": "integer",
            "description": "Step order number"
          },
          "instruction": {
            "type": "string",
            "description": "What the seller needs to do in this step"
          },
          "link": {
            "type": "string",
            "format": "uri",
            "description": "Optional URL to the form or page for this step"
          }
        }
      }
    },
    "requirements": {
      "type": "array",
      "description": "List of documents or requirements needed for registration",
      "items": { "type": "string" }
    },
    "support_contact": {
      "type": "string",
      "description": "Optional support email or contact info"
    }
  }
}

---

Example Query 2:
"Top otome villainess anime 2025"

Example Response 2:
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "Top Otome Villainess Anime List Schema",
  "type": "object",
  "properties": {
    "anime_list": {
      "type": "array",
      "description": "List of top otome villainess anime titles",
      "items": {
        "type": "object",
        "required": ["title", "genre", "year"],
        "properties": {
          "title": {
            "type": "string",
            "description": "Title of the anime series"
          },
          "genre": {
            "type": "string",
            "description": "Primary genre, e.g. otome, romance, fantasy"
          },
          "year": {
            "type": "string",
            "description": "Release year of the anime"
          },
          "synopsis": {
            "type": "string",
            "description": "Brief plot summary or theme"
          },
          "studio": {
            "type": "string",
            "description": "Name of the animation studio"
          },
          "episodes": {
            "type": "integer",
            "description": "Total number of episodes"
          },
          "rating": {
            "type": "number",
            "description": "Average viewer rating out of 10"
          },
          "villainess_name": {
            "type": "string",
            "description": "Name of the main villainess character"
          },
          "watch_links": {
            "type": "array",
            "description": "Links to platforms where the anime can be watched",
            "items": {
              "type": "string",
              "format": "uri"
            }
          }
        }
      }
    }
  },
  "required": ["anime_list"]
}
""".strip()


# Main function to demonstrate sample usage
if __name__ == "__main__":
    model = "gemma3:4b"

    # query = "Query: Philippines TikTok online seller registration steps 2025"
    query = "Query: Top otome villainess anime 2025"

    # generator = Ollama()
    # result = generator.generate(prompt)
    # print("Generated Output:")
    # print(result["output"])

    response = chat(query, model, system=SYSTEM)

    logger.info("DONE!")
