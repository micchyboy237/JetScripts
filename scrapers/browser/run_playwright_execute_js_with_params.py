from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)  # headless=True in CI
    page = browser.new_page()
    page.goto("https://trytestingthis.netlify.app/")

    # ────────────────────────────────────────────────
    # 1. Trigger alert with custom message via JS eval
    # ────────────────────────────────────────────────
    custom_message = "Hello from Python Playwright!"
    page.evaluate(
        """(msg) => {
            alert(msg);
        }""",
        custom_message,
    )

    # (page will show alert → you can manually dismiss it during demo)

    # ────────────────────────────────────────────────
    # 2. Get page title + custom text (multiple args as array)
    # ────────────────────────────────────────────────
    result = page.evaluate(
        """([prefix, suffix]) => {
            return prefix + document.title + suffix;
        }""",
        ["Page title is: ", " !!!"],
    )

    print(result)  # → "Page title is: Try Testing This !!!"

    # ────────────────────────────────────────────────
    # 3. Pass dict/object as argument (very clean & common)
    # ────────────────────────────────────────────────
    user_info = {"name": "Alice", "age": 28, "is_active": True}

    output = page.evaluate(
        """(user) => {
            return `User ${user.name} (${user.age}) is ${user.is_active ? 'active' : 'inactive'}`;
        }""",
        user_info,
    )

    print(output)  # → "User Alice (28) is active"

    # ────────────────────────────────────────────────
    # 7. Evaluate on a specific element (via locator)
    #    Example: change text of the double-click area
    #    → Moved here so it runs BEFORE login/navigation
    # ────────────────────────────────────────────────
    page.locator("text=Double-click me").evaluate(
        """(el, newText, color) => {
            el.textContent = newText;
            el.style.backgroundColor = color;
        }""",
        ["Double-click SUCCESS!", "#d4edda"],
    )

    # Optional: pause briefly so you can see the change in the browser
    page.wait_for_timeout(2500)

    # ────────────────────────────────────────────────
    # 4. Call existing function or manipulate DOM with args
    # ────────────────────────────────────────────────
    page.evaluate("""() => {
        window.myCustomLog = (level, message) => console.log(`[${level}] ${message}`);
    }""")

    page.evaluate(
        """([lvl, msg]) => {
            window.myCustomLog(lvl, msg);
        }""",
        ["INFO", "Login attempt started"],
    )

    # ────────────────────────────────────────────────
    # 5. Fill username/password via JS + submit (with args)
    #    Useful when .fill() / .click() is not enough or you want pure JS
    # ────────────────────────────────────────────────
    credentials = {"username": "test", "password": "test"}

    with page.expect_navigation(wait_until="domcontentloaded", timeout=15000):
        page.evaluate(
            """(creds) => {
                const uname = document.querySelector('#uname');
                const pwd  = document.querySelector('#pwd');
                const submit = document.querySelector('[type="submit"]') ||
                               document.querySelector('input[type="submit"]') ||
                               document.querySelector('button[type="submit"]');

                if (uname) uname.value = creds.username;
                if (pwd)   pwd.value   = creds.password;
                if (submit) submit.click();
                else console.error("No submit button found");
            }""",
            credentials,
        )

    # After this block the page has navigated → it's now safe to interact again
    print("After login → current URL:", page.url)
    print("New page title:", page.title())

    # ────────────────────────────────────────────────
    # 6. Run async JS with parameter (e.g. fetch simulation)
    # ────────────────────────────────────────────────
    status = page.evaluate(
        """async (url) => {
            try {
                const res = await fetch(url);
                return res.status;
            } catch (e) {
                return 'Error: ' + e.message;
            }
        }""",
        "https://httpbin.org/status/200",
    )

    print(status)  # → 200

    # Optional: If you prefer native Playwright for login instead of JS evaluate:
    # page.fill("#uname", "test")
    # page.fill("#pwd", "test")
    # page.click('[type="submit"]')
    # page.wait_for_load_state("domcontentloaded")

    browser.close()
