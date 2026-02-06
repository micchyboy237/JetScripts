html = """
<html>
    <head>
        <title>Sample Page</title>
        <style>.hidden{display:none}</style>
        <script>console.log("noise")</script>
    </head>
    <body>
        <nav>Home | About | Contact</nav>
        <article>
            <h1>Extracting Text from HTML</h1>
            <p>This is the first sentence.</p>
            <p>This pipeline removes boilerplate and splits sentences correctly.</p>
            <p>It works well for scraped content!</p>
        </article>
        <footer>© 2026 Example Corp</footer>
    </body>
</html>
"""

import trafilatura

text = trafilatura.extract(html)
print(f"\nHTML Text:\n{text}")
