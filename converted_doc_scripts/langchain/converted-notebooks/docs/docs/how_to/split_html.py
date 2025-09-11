from bs4 import Tag
from jet.logger import logger
from langchain.agents import AgentExecutor
from langchain_text_splitters import HTMLHeaderTextSplitter
from langchain_text_splitters import HTMLSectionSplitter
from langchain_text_splitters import HTMLSemanticPreservingSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger.basicConfig(filename=log_file)
logger.info(f"Logs: {log_file}")

PERSIST_DIR = f"{OUTPUT_DIR}/chroma"
os.makedirs(PERSIST_DIR, exist_ok=True)

"""
# How to split HTML

Splitting HTML documents into manageable chunks is essential for various text processing tasks such as natural language processing, search indexing, and more. In this guide, we will explore three different text splitters provided by LangChain that you can use to split HTML content effectively:

- [**HTMLHeaderTextSplitter**](#using-htmlheadertextsplitter)
- [**HTMLSectionSplitter**](#using-htmlsectionsplitter)
- [**HTMLSemanticPreservingSplitter**](#using-htmlsemanticpreservingsplitter)

Each of these splitters has unique features and use cases. This guide will help you understand the differences between them, why you might choose one over the others, and how to use them effectively.

```
%pip install -qU langchain-text-splitters
```

## Overview of the Splitters

### [HTMLHeaderTextSplitter](#using-htmlheadertextsplitter)

:::info
Useful when you want to preserve the hierarchical structure of a document based on its headings.
:::

**Description**: Splits HTML text based on header tags (e.g., `<h1>`, `<h2>`, `<h3>`, etc.), and adds metadata for each header relevant to any given chunk.

**Capabilities**:
- Splits text at the HTML element level.
- Preserves context-rich information encoded in document structures.
- Can return chunks element by element or combine elements with the same metadata.

___

### [HTMLSectionSplitter](#using-htmlsectionsplitter)

:::info 
Useful when you want to split HTML documents into larger sections, such as `<section>`, `<div>`, or custom-defined sections. 
:::

**Description**: Similar to HTMLHeaderTextSplitter but focuses on splitting HTML into sections based on specified tags.

**Capabilities**:
- Uses XSLT transformations to detect and split sections.
- Internally uses `RecursiveCharacterTextSplitter` for large sections.
- Considers font sizes to determine sections.
___

### [HTMLSemanticPreservingSplitter](#using-htmlsemanticpreservingsplitter)

:::info 
Ideal when you need to ensure that structured elements are not split across chunks, preserving contextual relevancy. 
:::

**Description**: Splits HTML content into manageable chunks while preserving the semantic structure of important elements like tables, lists, and other HTML components.

**Capabilities**:
- Preserves tables, lists, and other specified HTML elements.
- Allows custom handlers for specific HTML tags.
- Ensures that the semantic meaning of the document is maintained.
- Built in normalization & stopword removal

___

### Choosing the Right Splitter

- **Use `HTMLHeaderTextSplitter` when**: You need to split an HTML document based on its header hierarchy and maintain metadata about the headers.
- **Use `HTMLSectionSplitter` when**: You need to split the document into larger, more general sections, possibly based on custom tags or font sizes.
- **Use `HTMLSemanticPreservingSplitter` when**: You need to split the document into chunks while preserving semantic elements like tables and lists, ensuring that they are not split and that their context is maintained.

| Feature | HTMLHeaderTextSplitter | HTMLSectionSplitter | HTMLSemanticPreservingSplitter |
|--------------------------------------------|------------------------|---------------------|-------------------------------|
| Splits based on headers | Yes | Yes | Yes |
| Preserves semantic elements (tables, lists) | No | No | Yes |
| Adds metadata for headers | Yes | Yes | Yes |
| Custom handlers for HTML tags | No | No | Yes |
| Preserves media (images, videos) | No | No | Yes |
| Considers font sizes | No | Yes | No |
| Uses XSLT transformations | No | Yes | No |

## Example HTML Document

Let's use the following HTML document as an example:
"""
logger.info("# How to split HTML")

html_string = """
<!DOCTYPE html>
  <html lang='en'>
  <head>
    <meta charset='UTF-8'>
    <meta name='viewport' content='width=device-width, initial-scale=1.0'>
    <title>Fancy Example HTML Page</title>
  </head>
  <body>
    <h1>Main Title</h1>
    <p>This is an introductory paragraph with some basic content.</p>

    <h2>Section 1: Introduction</h2>
    <p>This section introduces the topic. Below is a list:</p>
    <ul>
      <li>First item</li>
      <li>Second item</li>
      <li>Third item with <strong>bold text</strong> and <a href='#'>a link</a></li>
    </ul>

    <h3>Subsection 1.1: Details</h3>
    <p>This subsection provides additional details. Here's a table:</p>
    <table border='1'>
      <thead>
        <tr>
          <th>Header 1</th>
          <th>Header 2</th>
          <th>Header 3</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>Row 1, Cell 1</td>
          <td>Row 1, Cell 2</td>
          <td>Row 1, Cell 3</td>
        </tr>
        <tr>
          <td>Row 2, Cell 1</td>
          <td>Row 2, Cell 2</td>
          <td>Row 2, Cell 3</td>
        </tr>
      </tbody>
    </table>

    <h2>Section 2: Media Content</h2>
    <p>This section contains an image and a video:</p>
      <img src='example_image_link.mp4' alt='Example Image'>
      <video controls width='250' src='example_video_link.mp4' type='video/mp4'>
      Your browser does not support the video tag.
    </video>

    <h2>Section 3: Code Example</h2>
    <p>This section contains a code block:</p>
    <pre><code data-lang="html">
    &lt;div&gt;
      &lt;p&gt;This is a paragraph inside a div.&lt;/p&gt;
    &lt;/div&gt;
    </code></pre>

    <h2>Conclusion</h2>
    <p>This is the conclusion of the document.</p>
  </body>
  </html>
"""

"""
## Using HTMLHeaderTextSplitter

[HTMLHeaderTextSplitter](https://python.langchain.com/api_reference/text_splitters/html/langchain_text_splitters.html.HTMLHeaderTextSplitter.html) is a "structure-aware" [text splitter](/docs/concepts/text_splitters/) that splits text at the HTML element level and adds metadata for each header "relevant" to any given chunk. It can return chunks element by element or combine elements with the same metadata, with the objectives of (a) keeping related text grouped (more or less) semantically and (b) preserving context-rich information encoded in document structures. It can be used with other text splitters as part of a chunking pipeline.

It is analogous to the [MarkdownHeaderTextSplitter](/docs/how_to/markdown_header_metadata_splitter) for markdown files.

To specify what headers to split on, specify `headers_to_split_on` when instantiating `HTMLHeaderTextSplitter` as shown below.
"""
logger.info("## Using HTMLHeaderTextSplitter")


headers_to_split_on = [
    ("h1", "Header 1"),
    ("h2", "Header 2"),
    ("h3", "Header 3"),
]

html_splitter = HTMLHeaderTextSplitter(headers_to_split_on)
html_header_splits = html_splitter.split_text(html_string)
html_header_splits

"""
To return each element together with their associated headers, specify `return_each_element=True` when instantiating `HTMLHeaderTextSplitter`:
"""
logger.info("To return each element together with their associated headers, specify `return_each_element=True` when instantiating `HTMLHeaderTextSplitter`:")

html_splitter = HTMLHeaderTextSplitter(
    headers_to_split_on,
    return_each_element=True,
)
html_header_splits_elements = html_splitter.split_text(html_string)

"""
Comparing with the above, where elements are aggregated by their headers:
"""
logger.info("Comparing with the above, where elements are aggregated by their headers:")

for element in html_header_splits[:2]:
    logger.debug(element)

"""
Now each element is returned as a distinct `Document`:
"""
logger.info("Now each element is returned as a distinct `Document`:")

for element in html_header_splits_elements[:3]:
    logger.debug(element)

"""
### How to split from a URL or HTML file:

To read directly from a URL, pass the URL string into the `split_text_from_url` method.

Similarly, a local HTML file can be passed to the `split_text_from_file` method.
"""
logger.info("### How to split from a URL or HTML file:")

url = "https://plato.stanford.edu/entries/goedel/"

headers_to_split_on = [
    ("h1", "Header 1"),
    ("h2", "Header 2"),
    ("h3", "Header 3"),
    ("h4", "Header 4"),
]

html_splitter = HTMLHeaderTextSplitter(headers_to_split_on)

html_header_splits = html_splitter.split_text_from_url(url)

"""
### How to constrain chunk sizes:

`HTMLHeaderTextSplitter`, which splits based on HTML headers, can be composed with another splitter which constrains splits based on character lengths, such as `RecursiveCharacterTextSplitter`.

This can be done using the `.split_documents` method of the second splitter:
"""
logger.info("### How to constrain chunk sizes:")


chunk_size = 500
chunk_overlap = 30
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size, chunk_overlap=chunk_overlap
)

splits = text_splitter.split_documents(html_header_splits)
splits[80:85]

"""
### Limitations

There can be quite a bit of structural variation from one HTML document to another, and while `HTMLHeaderTextSplitter` will attempt to attach all "relevant" headers to any given chunk, it can sometimes miss certain headers. For example, the algorithm assumes an informational hierarchy in which headers are always at nodes "above" associated text, i.e. prior siblings, ancestors, and combinations thereof. In the following news article (as of the writing of this document), the document is structured such that the text of the top-level headline, while tagged "h1", is in a *distinct* subtree from the text elements that we'd expect it to be *"above"*&mdash;so we can observe that the "h1" element and its associated text do not show up in the chunk metadata (but, where applicable, we do see "h2" and its associated text):
"""
logger.info("### Limitations")

url = "https://www.cnn.com/2023/09/25/weather/el-nino-winter-us-climate/index.html"

headers_to_split_on = [
    ("h1", "Header 1"),
    ("h2", "Header 2"),
]

html_splitter = HTMLHeaderTextSplitter(headers_to_split_on)
html_header_splits = html_splitter.split_text_from_url(url)
logger.debug(html_header_splits[1].page_content[:500])

"""
## Using HTMLSectionSplitter

Similar in concept to the [HTMLHeaderTextSplitter](#using-htmlheadertextsplitter), the `HTMLSectionSplitter` is a "structure-aware" [text splitter](/docs/concepts/text_splitters/) that splits text at the element level and adds metadata for each header "relevant" to any given chunk. It lets you split HTML by sections.

It can return chunks element by element or combine elements with the same metadata, with the objectives of (a) keeping related text grouped (more or less) semantically and (b) preserving context-rich information encoded in document structures.

Use `xslt_path` to provide an absolute path to transform the HTML so that it can detect sections based on provided tags. The default is to use the `converting_to_header.xslt` file in the `data_connection/document_transformers` directory. This is for converting the html to a format/layout that is easier to detect sections. For example, `span` based on their font size can be converted to header tags to be detected as a section.

### How to split HTML strings:
"""
logger.info("## Using HTMLSectionSplitter")


headers_to_split_on = [
    ("h1", "Header 1"),
    ("h2", "Header 2"),
]

html_splitter = HTMLSectionSplitter(headers_to_split_on)
html_header_splits = html_splitter.split_text(html_string)
html_header_splits

"""
### How to constrain chunk sizes:

`HTMLSectionSplitter` can be used with other text splitters as part of a chunking pipeline. Internally, it uses the `RecursiveCharacterTextSplitter` when the section size is larger than the chunk size. It also considers the font size of the text to determine whether it is a section or not based on the determined font size threshold.
"""
logger.info("### How to constrain chunk sizes:")


headers_to_split_on = [
    ("h1", "Header 1"),
    ("h2", "Header 2"),
    ("h3", "Header 3"),
]

html_splitter = HTMLSectionSplitter(headers_to_split_on)

html_header_splits = html_splitter.split_text(html_string)

chunk_size = 50
chunk_overlap = 5
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size, chunk_overlap=chunk_overlap
)

splits = text_splitter.split_documents(html_header_splits)
splits

"""
## Using HTMLSemanticPreservingSplitter

The `HTMLSemanticPreservingSplitter` is designed to split HTML content into manageable chunks while preserving the semantic structure of important elements like tables, lists, and other HTML components. This ensures that such elements are not split across chunks, causing loss of contextual relevancy such as table headers, list headers etc.

This splitter is designed at its heart, to create contextually relevant chunks. General Recursive splitting with `HTMLHeaderTextSplitter` can cause tables, lists and other structured elements to be split in the middle, losing significant context and creating bad chunks.

The `HTMLSemanticPreservingSplitter` is essential for splitting HTML content that includes structured elements like tables and lists, especially when it's critical to preserve these elements intact. Additionally, its ability to define custom handlers for specific HTML tags makes it a versatile tool for processing complex HTML documents.

**IMPORTANT**: `max_chunk_size` is not a definite maximum size of a chunk, the calculation of max size, occurs when the preserved content is not apart of the chunk, to ensure it is not split. When we add the preserved data back in to the chunk, there is a chance the chunk size will exceed the `max_chunk_size`. This is crucial to ensure we maintain the structure of the original document

:::info 

Notes:

1. We have defined a custom handler to re-format the contents of code blocks
2. We defined a deny list for specific html elements, to decompose them and their contents pre-processing
3. We have intentionally set a small chunk size to demonstrate the non-splitting of elements

:::
"""
logger.info("## Using HTMLSemanticPreservingSplitter")


headers_to_split_on = [
    ("h1", "Header 1"),
    ("h2", "Header 2"),
]


def code_handler(element: Tag) -> str:
    data_lang = element.get("data-lang")
    code_format = f"<code:{data_lang}>{element.get_text()}</code>"

    return code_format


splitter = HTMLSemanticPreservingSplitter(
    headers_to_split_on=headers_to_split_on,
    separators=["\n\n", "\n", ". ", "! ", "? "],
    max_chunk_size=50,
    preserve_images=True,
    preserve_videos=True,
    elements_to_preserve=["table", "ul", "ol", "code"],
    denylist_tags=["script", "style", "head"],
    custom_handlers={"code": code_handler},
)

documents = splitter.split_text(html_string)
documents

"""
### Preserving Tables and Lists
In this example, we will demonstrate how the `HTMLSemanticPreservingSplitter` can preserve a table and a large list within an HTML document. The chunk size will be set to 50 characters to illustrate how the splitter ensures that these elements are not split, even when they exceed the maximum defined chunk size.
"""
logger.info("### Preserving Tables and Lists")


html_string = """
<!DOCTYPE html>
<html>
    <body>
        <div>
            <h1>Section 1</h1>
            <p>This section contains an important table and list that should not be split across chunks.</p>
            <table>
                <tr>
                    <th>Item</th>
                    <th>Quantity</th>
                    <th>Price</th>
                </tr>
                <tr>
                    <td>Apples</td>
                    <td>10</td>
                    <td>$1.00</td>
                </tr>
                <tr>
                    <td>Oranges</td>
                    <td>5</td>
                    <td>$0.50</td>
                </tr>
                <tr>
                    <td>Bananas</td>
                    <td>50</td>
                    <td>$1.50</td>
                </tr>
            </table>
            <h2>Subsection 1.1</h2>
            <p>Additional text in subsection 1.1 that is separated from the table and list.</p>
            <p>Here is a detailed list:</p>
            <ul>
                <li>Item 1: Description of item 1, which is quite detailed and important.</li>
                <li>Item 2: Description of item 2, which also contains significant information.</li>
                <li>Item 3: Description of item 3, another item that we don't want to split across chunks.</li>
            </ul>
        </div>
    </body>
</html>
"""

headers_to_split_on = [("h1", "Header 1"), ("h2", "Header 2")]

splitter = HTMLSemanticPreservingSplitter(
    headers_to_split_on=headers_to_split_on,
    max_chunk_size=50,
    elements_to_preserve=["table", "ul"],
)

documents = splitter.split_text(html_string)
logger.debug(documents)

"""
#### Explanation
In this example, the `HTMLSemanticPreservingSplitter` ensures that the entire table and the unordered list (`<ul>`) are preserved within their respective chunks. Even though the chunk size is set to 50 characters, the splitter recognizes that these elements should not be split and keeps them intact.

This is particularly important when dealing with data tables or lists, where splitting the content could lead to loss of context or confusion. The resulting `Document` objects retain the full structure of these elements, ensuring that the contextual relevance of the information is maintained.

### Using a Custom Handler
The `HTMLSemanticPreservingSplitter` allows you to define custom handlers for specific HTML elements. Some platforms, have custom HTML tags that are not natively parsed by `BeautifulSoup`, when this occurs, you can utilize custom handlers to add the formatting logic easily. 

This can be particularly useful for elements that require special processing, such as `<iframe>` tags or specific 'data-' elements. In this example, we'll create a custom handler for `iframe` tags that converts them into Markdown-like links.
"""
logger.info("#### Explanation")

def custom_iframe_extractor(iframe_tag):
    iframe_src = iframe_tag.get("src", "")
    return f"[iframe:{iframe_src}]({iframe_src})"


splitter = HTMLSemanticPreservingSplitter(
    headers_to_split_on=headers_to_split_on,
    max_chunk_size=50,
    separators=["\n\n", "\n", ". "],
    elements_to_preserve=["table", "ul", "ol"],
    custom_handlers={"iframe": custom_iframe_extractor},
)

html_string = """
<!DOCTYPE html>
<html>
    <body>
        <div>
            <h1>Section with Iframe</h1>
            <iframe src="https://example.com/embed"></iframe>
            <p>Some text after the iframe.</p>
            <ul>
                <li>Item 1: Description of item 1, which is quite detailed and important.</li>
                <li>Item 2: Description of item 2, which also contains significant information.</li>
                <li>Item 3: Description of item 3, another item that we don't want to split across chunks.</li>
            </ul>
        </div>
    </body>
</html>
"""

documents = splitter.split_text(html_string)
logger.debug(documents)

"""
#### Explanation
In this example, we defined a custom handler for `iframe` tags that converts them into Markdown-like links. When the splitter processes the HTML content, it uses this custom handler to transform the `iframe` tags while preserving other elements like tables and lists. The resulting `Document` objects show how the iframe is handled according to the custom logic you provided.

**Important**: When presvering items such as links, you should be mindful not to include `.` in your separators, or leave separators blank. `RecursiveCharacterTextSplitter` splits on full stop, which will cut links in half. Ensure you provide a separator list with `. `  instead.

### Using a custom handler to analyze an image with an LLM

With custom handler's, we can also override the default processing for any element. A great example of this, is inserting semantic analysis of an image within a document, directly in the chunking flow.

Since our function is called when the tag is discovered, we can override the `<img>` tag and turn off `preserve_images` to insert any content we would like to embed in our chunks.

```python
"""
logger.info("#### Explanation")This example assumes you have helper methods `load_image_from_url` and an LLM agent `llm` that can process image data."""


# This example needs to be replaced with your own agent
llm = AgentExecutor(...)


# This method is a placeholder for loading image data from a URL and is not implemented here
def load_image_from_url(image_url: str) -> bytes:
    # Assuming this method fetches the image data from the URL
    return b"image_data"


html_string = """
logger.info("# This example needs to be replaced with your own agent")
<!DOCTYPE html>
<html>
    <body>
        <div>
            <h1>Section with Image and Link</h1>
            <p>
                <img src="https://example.com/image.jpg" alt="An example image" />
                Some text after the image.
            </p>
            <ul>
                <li>Item 1: Description of item 1, which is quite detailed and important.</li>
                <li>Item 2: Description of item 2, which also contains significant information.</li>
                <li>Item 3: Description of item 3, another item that we don't want to split across chunks.</li>
            </ul>
        </div>
    </body>
</html>
"""


def custom_image_handler(img_tag) -> str:
    img_src = img_tag.get("src", "")
    img_alt = img_tag.get("alt", "No alt text provided")

    image_data = load_image_from_url(img_src)
    semantic_meaning = llm.invoke(image_data)

    markdown_text = f"[Image Alt Text: {img_alt} | Image Source: {img_src} | Image Semantic Meaning: {semantic_meaning}]"

    return markdown_text


splitter = HTMLSemanticPreservingSplitter(
    headers_to_split_on=headers_to_split_on,
    max_chunk_size=50,
    separators=["\n\n", "\n", ". "],
    elements_to_preserve=["ul"],
    preserve_images=False,
    custom_handlers={"img": custom_image_handler},
)

documents = splitter.split_text(html_string)

logger.debug(documents)
```

```
[Document(metadata={'Header 1': 'Section with Image and Link'}, page_content='[Image Alt Text: An example image | Image Source: https://example.com/image.jpg | Image Semantic Meaning: semantic-meaning] Some text after the image'), 
Document(metadata={'Header 1': 'Section with Image and Link'}, page_content=". Item 1: Description of item 1, which is quite detailed and important. Item 2: Description of item 2, which also contains significant information. Item 3: Description of item 3, another item that we don't want to split across chunks.")]
```

#### Explanation:

With our custom handler written to extract the specific fields from a `<img>` element in HTML, we can further process the data with our agent, and insert the result directly into our chunk. It is important to ensure `preserve_images` is set to `False` otherwise the default processing of `<img>` fields will take place.
"""
logger.info("#### Explanation:")


logger.info("\n\n[DONE]", bright=True)