

# Example Usage
import os
import shutil
from jet.file.utils import load_file, save_file
from jet.logger import logger
from jet.scrapers.utils import extract_by_heading_hierarchy, extract_texts_by_hierarchy, extract_tree_with_text, extract_text_elements, format_html, print_html
from jet.search.formatters import clean_string
from jet.utils.commands import copy_to_clipboard


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
    from jet.scrapers.preprocessor import html_to_markdown

    data_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/searched_html_myanimelist_net_Isekai/doc.html"
    output_dir = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/scrapers/generated/run_format_html"

    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)

    html_doc: str = load_file(data_file)

    save_file(html_doc, f"{output_dir}/doc.html")

    # Headings
    headings = extract_texts_by_hierarchy(html_doc)
    save_file(headings, f"{output_dir}/headings.json")

    texts = [item["text"] for item in headings]
    md_text = "\n\n".join(texts)
    save_file(md_text, f"{output_dir}/md_text.md")

    # By headings
    header_elements = extract_by_heading_hierarchy(html_doc)
    save_file(header_elements, f"{output_dir}/headings_elements.json")

    header_texts = []
    for idx, node in enumerate(header_elements):
        texts = [
            f"Document {idx + 1} | Tag ({node.tag}) | Depth ({node.depth})"
        ]
        if node.parent:
            texts.append(f"Parent ({node.parent})")

        child_texts = [child_node.text or " " for child_node in node.children]

        texts.extend([
            "Text:",
            node.text + "\n" + ''.join(child_texts)
        ])
        header_texts.append("\n".join(texts))
    save_file("\n\n---\n\n".join(header_texts), f"{output_dir}/headings.md")

    # Get the tree-like structure
    tree_elements = extract_tree_with_text(html_doc)
    save_file(tree_elements, f"{output_dir}/tree_elements.json")

    # text_elements = extract_text_elements(html_doc)
    # save_file(text_elements, f"{output_dir}/text_elements.json")

    formatted_html = format_html(html_doc)
    save_file(formatted_html, f"{output_dir}/formatted_html.html")

    print_html(html_doc)
