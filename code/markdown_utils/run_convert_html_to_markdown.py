import os

from jet.code.markdown_utils._converters import convert_html_to_markdown
from jet.file.utils import save_file

html_1 = """
<p>
 Sample title
</p>
<h1 id="project-overview">
 Project Overview
</h1>
<p>
 Welcome to our
 <strong>
  project
 </strong>
 ! This is an
 <code>
  introduction
 </code>
 to our work, featuring a
 <a href="https://project.com">
  website
 </a>
 .
</p>
<p>
 <img alt="Project Logo" src="https://project.com/logo.png"/>
</p>
<blockquote>
 <p>
  <strong>
   Note
  </strong>
  : Always check the
  <a href="https://docs.project.com">
   docs
  </a>
  for updates.
 </p>
</blockquote>
<h2 id="features">
 Features
</h2>
<ul>
 <li>
  [ ] Task 1: Implement login
 </li>
 <li>
  [x] Task 2: Add dashboard
 </li>
 <li>
  Task 3: Optimize performance
 </li>
</ul>
<h3 id="technical-details">
 Technical Details
</h3>
<pre><code class="language-python">def greet(name: str) -&gt; str:
    return f"Hello, {name}!"
</code></pre>
<h4 id="api-endpoints">
 API Endpoints
</h4>
<table>
 <thead>
  <tr>
   <th>
    Endpoint
   </th>
   <th>
    Method
   </th>
   <th>
    Description
   </th>
  </tr>
 </thead>
 <tbody>
  <tr>
   <td>
    /api/users
   </td>
   <td>
    GET
   </td>
   <td>
    Fetch all users
   </td>
  </tr>
  <tr>
   <td>
    /api/users/{id}
   </td>
   <td>
    POST
   </td>
   <td>
    Create a new user
   </td>
  </tr>
 </tbody>
</table>
<h5 id="inline-code">
 Inline Code
</h5>
<p>
 Use
 <code>
  print("Hello")
 </code>
 for quick debugging.
</p>
<h6 id="emphasis">
 Emphasis
</h6>
<p>
 <em>
  Italic
 </em>
 ,
 <strong>
  bold
 </strong>
 , and
 <strong>
  <em>
   bold italic
  </em>
 </strong>
 text are supported.
</p>
<div class="alert">
 This is an HTML block.
</div>
<p>
 <span class="badge">
  New
 </span>
 inline HTML.
</p>
<p>
 [^1]: This is a footnote reference.
[^1]: Footnote definition here.
</p>
<h2 id="unordered-list">
 Unordered list
</h2>
<ul>
 <li>
  List item 1
  <ul>
   <li>
    Nested item
   </li>
  </ul>
 </li>
 <li>
  List item 2
 </li>
 <li>
  List item 3
 </li>
</ul>
<h2 id="ordered-list">
 Ordered list
</h2>
<ol>
 <li>
  Ordered item 1
 </li>
 <li>
  Ordered item 2
 </li>
 <li>
  Ordered item 3
 </li>
</ol>
<h2 id="inline-html">
 Inline HTML
</h2>
<p>
 <span class="badge">
  New
 </span>
 inline HTML
</p>
"""

html_2 = """
<pre><code class="language-python">def hello():
    print("Hello, World!")
</code></pre>
<table>
 <thead>
  <tr>
   <th>
    Header1
   </th>
   <th>
    Header2
   </th>
  </tr>
 </thead>
 <tbody>
  <tr>
   <td>
    Cell1
   </td>
   <td>
    Cell2
   </td>
  </tr>
 </tbody>
</table>
<p>
 A paragraph with custom attributes {#para1 .class1 style="color: blue;"}
</p>
<dl>
 <dt>
  Term 1
 </dt>
 <dd>
  Definition for term 1.
 </dd>
 <dt>
  Term 2
 </dt>
 <dd>
  Definition for term 2.
 </dd>
</dl>
<pre><code class="language-javascript">function greet() {
    console.log("Hello!");
}
</code></pre>
<p>
 Here is some text[^1].
</p>
<p>
 [^1]: This is a footnote.
</p>
<div markdown="1">
 *Emphasis* inside a div.
</div>
<pre><code class="language-python">def example():
    pass
</code></pre>
<table>
 <thead>
  <tr>
   <th>
    Col1
   </th>
   <th>
    Col2
   </th>
  </tr>
 </thead>
 <tbody>
  <tr>
   <td>
    A
   </td>
   <td>
    B
   </td>
  </tr>
 </tbody>
</table>
<p>
 Text with footnote[^1].
</p>
<p>
 [^1]: Footnote content.
</p>
<p>
 LOL and WTF are abbreviations.
</p>
<p>
 <em>
  [LOL]: Laughing Out Loud
 </em>
 [WTF]: What The Fudge
</p>
<pre><code class="language-python">def example():
    print('Hello')
</code></pre>
<p>
 Paragraph with
 <em>
  legacy
 </em>
 attributes {id="my-id" class="my-class"}.
</p>
<p>
 <em>
  italic
 </em>
 and
 <strong>
  bold
 </strong>
 text.
</p>
<ul>
 <li>
  Item 1
  <ol>
   <li>
    Subitem A
   </li>
   <li>
    Subitem B
   </li>
  </ol>
 </li>
 <li>
  Item 2
 </li>
</ul>
<p>
 He said, "Hello..." and used -- and --- in text.
</p>
<div class="toc">
 <ul>
  <li>
   <a href="#heading-1">
    Heading 1
   </a>
   <ul>
    <li>
     <a href="#heading-2">
      Heading 2
     </a>
     <ul>
      <li>
       <a href="#heading-3">
        Heading 3
       </a>
      </li>
     </ul>
    </li>
   </ul>
  </li>
 </ul>
</div>
<h1 id="heading-1">
 Heading 1
</h1>
<h2 id="heading-2">
 Heading 2
</h2>
<h3 id="heading-3">
 Heading 3
</h3>
<p>
 This is a [[WikiLink]].
</p>
"""

if __name__ == "__main__":
    output_dir = os.path.join(os.path.dirname(
        __file__), "generated", os.path.splitext(os.path.basename(__file__))[0])

    converted_markdown_1 = convert_html_to_markdown(html_1)
    converted_markdown_1_no_links = convert_html_to_markdown(html_1, ignore_links=True)
    converted_markdown_2 = convert_html_to_markdown(html_2)
    converted_markdown_2_no_links = convert_html_to_markdown(html_2, ignore_links=True)

    save_file(converted_markdown_1, f"{output_dir}/converted_markdown_1.md")
    save_file(converted_markdown_1_no_links, f"{output_dir}/converted_markdown_1_no_links.md")
    save_file(converted_markdown_2, f"{output_dir}/converted_markdown_2.md")
    save_file(converted_markdown_2_no_links, f"{output_dir}/converted_markdown_2_no_links.md")
