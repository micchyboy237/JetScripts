from hrequests import get, Session, parser, BrowserSession, Proxy, async_requests

# Basic Request
resp = get("https://example.com")
print(resp.status_code, resp.text)

# HTML Parsing
html_parser = parser.HTML(resp.text)
links = html_parser.find("a")
for link in links:
    print(link.text(), link.attributes.get("href"))

# Browser Automation
with BrowserSession() as browser:
    browser.goto("https://example.com")
    print(browser.content())
    browser.screenshot("screenshot.png")

# Concurrent Requests
urls = ["https://example.com", "https://google.com"]
responses = async_requests([get(url) for url in urls])
for resp in responses:
    print(resp.status_code)

# Proxy Usage
proxy = Proxy("username", "api_key")
resp = get("https://example.com", proxies=proxy.get_proxy())
print(resp.text)
