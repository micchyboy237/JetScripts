from jet.utils.url_utils import clean_url


if __name__ == "__main__":
    urls_to_clean = [
        "https://thefilibusterblog.com/es/upcoming-isekai-anime-releases-for-2025-latest-announcements/",
        "https://peacedoorball.blog/en/upcoming-isekai-anime-releases-for-2025-all-announced-titles-so-far#",
        "https://fyuu.net/new-isekai-anime-2025#content",
    ]
    cleaned_urls = [clean_url(url) for url in urls_to_clean]
    for url in cleaned_urls:
        print("-", url)
