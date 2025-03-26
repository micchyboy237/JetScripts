import requests
import os

MAL_CLIENT_ID = os.getenv("MAL_CLIENT_ID")


def fetch_anime_details(anime_title):
    url = f"https://api.myanimelist.net/v2/anime?q={anime_title}&limit=1&fields=status,num_episodes,broadcast"
    headers = {"X-MAL-CLIENT-ID": MAL_CLIENT_ID}
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        data = response.json()["data"][0]["node"]
        return {
            "status": data.get("status", "Unknown"),
            "episodes": data.get("num_episodes", "Unknown"),
            "broadcast": data.get("broadcast", {}).get("string", "Unknown")
        }
    return None
