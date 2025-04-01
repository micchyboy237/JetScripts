import json
from typing import List, Dict, Optional
from jet.logger import logger
from pyquery import PyQuery as pq


# Define the structure of a tree node with TypedDict
class TreeNode(Dict):
    tag: str
    text: str
    depth: int
    children: List['TreeNode']  # Recursive reference to TreeNode


def find_parents_with_text(html: str) -> Optional[TreeNode]:
    """
    Finds all elements (including <p>, <a>, <h1-h6>) that contain any text, ensuring a natural document order.
    Returns a tree-like structure of parents and their corresponding text, including depth for each node.

    :param html: The HTML string to parse.
    :return: A tree-like structure with parent elements and their corresponding text.
    """
    # Helper function to recursively build the tree
    def build_tree(element, current_depth: int) -> Optional[TreeNode]:
        text = pq(element).text().strip()
        if text:  # If element contains text
            children = []
            for child in pq(element).children():  # Recursively process children
                child_tree = build_tree(child, current_depth + 1)
                if child_tree:
                    children.append(child_tree)

            return {
                # Tag of the element (e.g., div, p, h1)
                "tag": pq(element)[0].tag,
                "text": text,               # The text inside this element
                "depth": current_depth,     # Current depth in the hierarchy
                "children": children        # Nested children that also contain text
            }
        return None

    doc = pq(html)
    # Start with the root element (<html>) at depth 0
    root = build_tree(doc[0], 0)

    return root  # Returns tree-like structure starting from <html> element


# Example Usage
html_doc = """
<html>
  <head>
    <title>News Website</title>
  </head>
  <body>
    <header>
      <h1>Today's News</h1>
      <nav>
        <ul>
          <li><a href="#">Home</a></li>
          <li><a href="#">World</a></li>
          <li><a href="#">Politics</a></li>
          <li><a href="#">Business</a></li>
        </ul>
      </nav>
    </header>
    
    <main>
      <section class="top-stories">
        <article>
          <h2>Breaking: Major Event Unfolds</h2>
          <p>Details are still emerging about a major event happening right now.</p>
        </article>
      </section>

      <section class="latest-updates">
        <article>
          <h3>Economy Sees Growth</h3>
          <p>The latest reports show an unexpected increase in GDP this quarter.</p>
        </article>
        
        <article>
          <h3>Tech Innovations in 2025</h3>
          <p>New advancements in AI and automation are set to change industries.</p>
        </article>
      </section>
    </main>

    <aside>
      <h3>Sponsored Content</h3>
      <p>Check out this amazing product that is changing lives.</p>
    </aside>

    <footer>
      <p>© 2025 News Website. All rights reserved.</p>
    </footer>
  </body>
</html>
"""

# Get the tree-like structure
tree = find_parents_with_text(html_doc)

# Function to print the tree-like structure recursively


def print_tree(node: TreeNode, indent=0):
    if node:
        logger.log(('  ' * indent + f"{node['depth']}:"), "-", node['tag'], "-",
                   json.dumps(node['text'][:30]), colors=["INFO", "GRAY", "DEBUG", "GRAY", "SUCCESS"])
        for child in node['children']:
            print_tree(child, indent + 1)


# Print the tree structure
print_tree(tree)
