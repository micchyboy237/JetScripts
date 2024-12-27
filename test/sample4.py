import json
from jet.search import search_searxng
from jet.logger import logger

if __name__ == "__main__":
    results = search_searxng(
        query_url="http://searxng.local:8080/search",
        query="joe's greenwich village address",
        min_score=0,
        engines=["google"],
        use_cache=False,
    )
    logger.success(json.dumps(results, indent=2))
