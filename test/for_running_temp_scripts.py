import hrequests
from jet.logger import logger
from jet.scrapers.browser.playwright import PageContent, scrape_sync, setup_sync_browser_page
from jet.transformers.formatters import format_json
from jet.utils.commands import copy_to_clipboard
from shared.data_types.job import JobData
from tqdm import tqdm
from jet.file.utils import save_file, load_file

session = setup_sync_browser_page(headless=False)


def scrape_job_details(job_link: str) -> dict:
    wait_for_css = [
        ".description__text",
        ".description__job-criteria-item"
    ]
    page_content: PageContent = scrape_sync(
        job_link, wait_for_css, browser_page=session)
    html_content = page_content["html"]
    htmlParser = hrequests.HTML(html=html_content)

    header_element = htmlParser.find("h1")
    company_element = htmlParser.find(
        ".top-card-layout__second-subline .topcard__org-name-link")
    description_element = htmlParser.find(".description__text")
    info_elements: list[hrequests.parser.Element] = htmlParser.find_all(
        ".description__job-criteria-item")

    title = header_element.text.strip() if header_element else ""
    company = company_element.text.strip() if company_element else ""
    details = description_element.text.strip() if description_element else ""

    job_type = None
    for info_elm in info_elements:
        label = info_elm.find('h3').text.strip()
        value = info_elm.find('span').text.strip()

        if label.lower() == "employment type":
            job_type = value

    return {
        "title": title,
        "company": company,
        "job_type": job_type,
        "details": details
    }


def main():
    # Load job data
    data_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/my-jobs/saved/jobs.json"
    data: list[JobData] = load_file(data_file) or []

    failed_results = []

    for d in tqdm(data):
        title = d['title']
        company = d['company']

        if any(text.startswith("***") for text in [title, company]):
            failed_results.append(d)

    for d in failed_results:
        details = scrape_job_details(d['link'])
        for key, value in details.items():
            if d[key].startswith("***"):
                d.update({key: value})
        logger.success(format_json(d))

    save_file(data, data_file)


if __name__ == "__main__":
    main()
