

# Example Usage
from jet.file.utils import load_file
from jet.scrapers.utils import extract_tree_with_text, print_tree
from jet.search.formatters import clean_string


html_doc = """
<html>
  <head>
    <title>Global News Hub</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="styles.css">
  </head>
  <body>
    <header>
      <h1 id="main-heading">Welcome to Global News</h1>
      <nav>
        <ul>
          <li><a href="#">Home</a></li>
          <li><a href="#">World</a></li>
          <li><a href="#">Technology</a></li>
          <li><a href="#">Health</a></li>
          <li><a href="#">Sports</a></li>
          <li><a href="#">Entertainment</a></li>
        </ul>
      </nav>
    </header>
    
    <main>
      <section class="featured-stories">
        <h2>Featured Stories</h2>
        <article class="featured-article">
          <h3>Breaking: Earthquake Hits City</h3>
          <img src="earthquake.jpg" alt="Damaged buildings" />
          <p>A magnitude 7.8 earthquake has struck a major city causing widespread destruction.</p>
          <a href="#">Read More</a>
        </article>
      </section>

      <section class="latest-updates">
        <h2>Latest Updates</h2>
        <article>
          <h3>The Future of AI in Healthcare</h3>
          <p>AI technology is advancing rapidly, with a potential to revolutionize healthcare systems.</p>
        </article>

        <article>
          <h3>Climate Change: What Can We Do?</h3>
          <p>Experts discuss the steps needed to address climate change and its impact on the environment.</p>
        </article>

        <article>
          <h3>New Mars Rover Images Released</h3>
          <p>NASA's new rover has sent back stunning images of the Martian landscape.</p>
          <figure>
            <img src="mars_rover.jpg" alt="Mars surface" />
            <figcaption>New photo from Mars Rover</figcaption>
          </figure>
        </article>
      </section>

      <section class="technology-news">
        <h2>Tech Innovations</h2>
        <ul>
          <li><a href="#">AI Advancements in Robotics</a></li>
          <li><a href="#">5G Technology: What’s Next?</a></li>
          <li><a href="#">SpaceX: Launching the Future</a></li>
        </ul>
      </section>

      <aside class="advertisements">
        <h3>Sponsored Content</h3>
        <div class="ad-banner">
          <p>Ad: Check out the latest smartphone on sale today!</p>
        </div>
        <div class="ad-banner">
          <p>Ad: New fitness app to track your workouts – get it now!</p>
        </div>
      </aside>
    </main>

    <footer>
      <p>© 2025 Global News Hub. All rights reserved.</p>
      <p>Follow us on:
        <a href="#">Twitter</a> | 
        <a href="#">Facebook</a> | 
        <a href="#">Instagram</a>
      </p>
    </footer>
  </body>
</html>
"""

if __name__ == "__main__":
    data_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_python_modules/jet/scrapers/crawler/generated/crawl/reelgood.com_urls.json"
    data = load_file(data_file)
    docs = []
    for item in data:
        docs.append(item["html"])

    # Get the tree-like structure
    # tree = extract_tree_with_text(html_doc)
    tree = extract_tree_with_text(docs[0])
    # Print the tree structure
    print_tree(tree)
