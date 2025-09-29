from jet.logger import logger
from jet.scrapers.utils import extract_favicon_ico_link


if __name__ == "__main__":
    favicon_ico_link = extract_favicon_ico_link("https://gamerant.com/new-isekai-anime-2024-upcoming")
    if favicon_ico_link:
        logger.success(f"favicon_ico_link: {favicon_ico_link}")
    else:
        logger.error("Failed to extract favicon.ico link.")
