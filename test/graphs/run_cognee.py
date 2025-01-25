import cognee
from cognee.api.v1.search import SearchType


async def main():
    text = """Natural language processing (NLP) is an interdisciplinary
       subfield of computer science and information retrieval"""
    await cognee.add(text)  # Add a new piece of information
    await cognee.cognify()  # Use LLMs and cognee to create knowledge
    # Query cognee for the knowledge
    search_results = await cognee.search(SearchType.INSIGHTS, query_text='Tell me about NLP')
    for result_text in search_results:
        print(result_text)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main)
