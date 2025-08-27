import requests


def get_top_isekai_anime():
    url = "https://www.bing.com/search?q=top+isekai+anime+2025&FORM=QBLH"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
    response = requests.get(url, headers=headers)

    # Parse the HTML content of the webpage
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(response.text, 'html.parser')
    anime_titles = []

    for link in soup.find_all('a'):
        if 'top isekai anime' in link.text.lower():
            title = link.text.strip()
            anime_titles.append(title)

    return anime_titles


# Execute the function to get the top isekai anime titles
top_isekai_anime = get_top_isekai_anime()

for i, title in enumerate(top_isekai_anime):
    print(f"{i+1}. {title}")
