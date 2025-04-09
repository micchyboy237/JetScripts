from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    # Print the executable paths for each browser
    print("Chromium executable path:", p.chromium.executable_path)
    print("Firefox executable path:", p.firefox.executable_path)
    print("WebKit executable path:", p.webkit.executable_path)
