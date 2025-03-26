from retriever.search import hybrid_search


def retrieve_anime_info(query):
    results = hybrid_search(query)

    if results:
        anime_data = f"Title: {results[0][0]}\nSynopsis: {results[0][1]}\nStatus: {results[0][2]}\nEpisodes: {results[0][3]}\nAiring: {results[0][4]}\nSource: {results[0][5]}"
    else:
        anime_data = "No relevant anime found."

    return anime_data


if __name__ == "__main__":
    query = input("Enter your anime query: ")
    answer = retrieve_anime_info(query)
    print("\n🔹 AI Response:\n", answer)
