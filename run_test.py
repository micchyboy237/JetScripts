from jet.logger import logger
from jet.automation import SeleniumScraper
from parsel import Selector


def extract_sections(html_content: str):
    """
    Extract sections where an <h2> is immediately followed by a <blockquote>,
    capturing sibling elements until the next <h2>.
    Args:
        html_content (str): The HTML content as a string.
    Returns:
        list[dict]: A list of sections, each containing the <h2> text, <blockquote> text, and the section HTML.
    """
    selector = Selector(text=html_content)
    h2_elements = selector.css("h2")
    first_section = [h2 for h2 in h2_elements if h2.xpath(
        "following-sibling::blockquote[1]")][0]
    all_elements = [first_section] + \
        first_section.xpath("following-sibling::*")

    sections = []
    section = ""
    for element in all_elements:
        is_heading = element.root.tag.startswith('h')
        # No tags
        text = element.xpath("string()").get().strip()
        line = f"{text}\n"
        if is_heading:
            if section:
                sections.append(section.strip())
            section = line
        else:
            section += line
    if section:
        sections.append(section.strip())

    return sections


class UrlScraper():
    def __init__(self) -> None:
        self.scraper = SeleniumScraper()

    def scrape_url(self, url: str) -> str:
        self.scraper.navigate_to_url(url)
        html_str = self.scraper.get_page_source()
        return html_str


if __name__ == "__main__":
    url = "https://developer.todoist.com/rest/v2/?shell#overview"
    url_scraper = UrlScraper()
    html_str = url_scraper.scrape_url(url)

    sections = extract_sections(html_str)
    # Filter sections with curl commands
    sections = [section for section in sections if "$ curl" in section]
    for section_idx, section in enumerate(sections):
        print(f"Section {section_idx + 1}:\n{section}")
        print("----")
    with open("generated/scraped_contents.md", "w", encoding="UTF-8") as f:
        f.write("\n\n".join(sections))
    logger.log("Sections:", len(sections), colors=["LOG", "SUCCESS"])
    logger.success("generated/scraped_contents.md", bright=True)
