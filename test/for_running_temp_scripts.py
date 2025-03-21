from urllib.parse import quote
from typing import List, Optional


def construct_google_query(
    title: str,
    properties: Optional[List[str]] = None,
    site: Optional[str] = None,
    exclude: Optional[List[str]] = None
) -> str:
    """
    Constructs a structured Google search query URL.

    :param title: The title of the anime.
    :param properties: A list of properties to search for (e.g., seasons, episodes, synopsis).
    :param site: A specific website to search in (e.g., "myanimelist.net").
    :param exclude: A list of words to exclude from search results.
    :return: A Google search URL string.
    """
    query = f'"{title}" anime'

    if properties:
        query += " " + " ".join(properties)

    if site:
        query += f" site:{site}"

    if exclude:
        query += " " + " ".join(f"-{word}" for word in exclude)

    return f"https://www.google.com/search?q={quote(query)}"
