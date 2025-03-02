import hrequests
from jet.logger.timer import sleep_countdown


def find_job_subtitle(url):
    # Start a browser session (headless by default)
    # page = hrequests.BrowserSession()
    # page = hrequests.chrome.Session(os='mac')

    try:
        # Open the target URL
        # page.get(url)
        # sleep_countdown(2)

        page = hrequests.get(url)

        # Find the element using the class selector
        job_subtitle_element = page.find('.jobs-search-results-list__subtitle')

        if job_subtitle_element:
            return job_subtitle_element.text()
        else:
            return "Element not found"

    finally:
        pass
        # Close the browser session
        # page.close()


        # Example usage
if __name__ == "__main__":
    # Update this to the correct job search URL
    url = "https://ph.linkedin.com/jobs/search?currentJobId=4072902804&f_TPR=r1209600&f_WT=2&geoId=103121230&keywords=React%20Native&origin=JOB_SEARCH_PAGE_JOB_FILTER&refresh=true"
    result = find_job_subtitle(url)
    print("Job Subtitle:", result)
