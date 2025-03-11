import asyncio
import json
from jet.wordnet.similarity import get_similar_texts
from jet.logger import logger


async def main():
    # Mock memories list with unique and similar entries
    mock_memories = [
        "October seven is the date of our vacation to Camarines Sur.",
        'October 7 is our vacation for Camarines Sur.',
        'October 7 is not our vacation for Camarines Sur.',
    ]

    similar_memories = get_similar_texts(mock_memories)

    # Simulate getting similar memories
    logger.log("Similar Memories Results:")
    logger.success(similar_memories)


# Run the main function
if __name__ == "__main__":
    asyncio.run(main())
