from jet.utils.url_utils import get_all_sitemap_urls


if __name__ == "__main__":
    all_urls = get_all_sitemap_urls(
        "https://gamerant.com/new-isekai-anime-2025")
    print(f"\nFound {len(all_urls)} URLs:")
    for url in all_urls[:10]:  # just print first 10 for brevity
        print("-", url)
