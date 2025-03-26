from jet.llm.ollama.base import Ollama
from llama_index.core.base.llms.types import ChatResponse


def summarize_anime_info(stored_data, mal_data, anilist_data):
    prompt = f"""
    Summarize the following anime details:
    
    Stored Data: {stored_data}
    MyAnimeList API Data: {mal_data}
    AniList API Data: {anilist_data}
    """

    llm = Ollama(model="llama3.1", stream=True)
    response: ChatResponse = llm.chat(prompt)

    return response.message.content
