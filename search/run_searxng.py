import os
import json
from jet.utils.commands import copy_to_clipboard
import requests
from jet.search.searxng import search_searxng, SearchResult
from jet.logger import logger


# Change the current working directory to the script's directory
file_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(file_dir)


if __name__ == "__main__":
    # fields = ["seasons", "episodes"]
    # search_keys_str = ", ".join(
    #     [key.replace('.', ' ').replace('_', ' ') for key in fields])
    # title = "I'll Become a Villainess Who Goes Down in History"
    # query = f"Anime \"{title}\" {search_keys_str}"

    query = "Top isekai anime 2025"

    try:
        include_sites = [
            # "https://easypc.com.ph",
            # "9anime",
            # "zoro"
            # "aniwatch"
            "myanimelist.net",
            "reelgood.com",
        ]
        exclude_sites = ["wikipedia.org"]
        engines = [
            "google",
            "brave",
            "duckduckgo",
            "bing",
            "yahoo",
        ]
        results: list[SearchResult] = search_searxng(
            query_url="http://Jethros-MacBook-Air.local:3000/search",
            query=query,
            min_score=0.2,
            include_sites=include_sites,
            exclude_sites=exclude_sites,
            engines=engines,
            # config={
            #     "port": 3101
            # },
        )
        copy_to_clipboard(results)
        logger.success(json.dumps(results, indent=2))
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching search results: {e}")
