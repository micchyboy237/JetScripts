import os
import json
from jet.utils.commands import copy_to_clipboard
import requests
from jet.search.searxng import fetch_search_results, QueryResponse
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

    params = {
        "query_url": "http://jethros-macbook-air.local:3000/search?q=Top+isekai+anime+2025+site%3Amyanimelist.net+OR+site%3Areelgood.com+-site%3Awikipedia.org&format=json&pageno=1&safesearch=2&language=en&categories=general&engines=google%2Cbrave%2Cduckduckgo%2Cbing%2Cyahoo",
        "headers": {
            "Accept": "application/json"
        },
        "params": {
            "q": "Top isekai anime 2025 site:myanimelist.net OR site:reelgood.com -site:wikipedia.org",
            "format": "json",
            "pageno": 1,
            "safesearch": 2,
            "language": "en",
            "categories": "general",
            "engines": "google,brave,duckduckgo,bing,yahoo"
        }
    }

    try:
        response: QueryResponse = fetch_search_results(**params)
        copy_to_clipboard(response)
        logger.success(json.dumps(response, indent=2))
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching search results: {e}")
