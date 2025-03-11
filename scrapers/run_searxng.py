import os
import json
import requests
from jet.search import search_searxng
from jet.logger import logger


# Change the current working directory to the script's directory
file_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(file_dir)


if __name__ == "__main__":

    try:
        filter_sites = [
            # "https://easypc.com.ph",
            # "9anime",
            # "zoro"
            "aniwatch"
        ]
        engines = [
            "google",
            "brave",
            "duckduckgo",
            "bing",
            "yahoo",
            "duckduckgo",
        ]
        results = search_searxng(
            query_url="http://searxng.local:8080/search",
            query="How many seasons and episodes does ”I’ll Become a Villainess Who Goes Down in History” anime have?",
            min_score=0,
            filter_sites=filter_sites,
            engines=engines,
            config={
                "port": 3101
            },
        )
        logger.success(json.dumps(results, indent=2))
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching search results: {e}")
