import requests


def fetch_anilist_details(anime_title):
    url = "https://graphql.anilist.co"
    query = """
    query ($search: String) {
      Media(search: $search, type: ANIME) {
        title { romaji }
        nextAiringEpisode { episode airingAt }
      }
    }
    """
    variables = {"search": anime_title}

    response = requests.post(
        url, json={"query": query, "variables": variables})

    if response.status_code == 200:
        data = response.json()["data"]["Media"]
        return {
            "next_episode": data["nextAiringEpisode"]["episode"] if data["nextAiringEpisode"] else "Unknown",
            "airing_at": data["nextAiringEpisode"]["airingAt"] if data["nextAiringEpisode"] else "Unknown"
        }
    return None
