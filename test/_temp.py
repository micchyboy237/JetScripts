from pathlib import Path

html_content = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>HTML Elements Demo</title>
<style>
body { font-family: sans-serif; padding: 20px; line-height: 1.5; }
section { margin-bottom: 40px; }
code { background: #f0f0f0; padding: 2px 4px; }
pre { background: #f7f7f7; padding: 10px; overflow:auto; }
h2 { border-bottom: 1px solid #ccc; padding-bottom: 4px; }
</style>
</head>
<body>
<h1>HTML Elements Demonstration</h1>

<section>
<h2>Text & Inline Elements</h2>
<p><strong>strong</strong>, <em>em</em>, <u>u</u>, <mark>mark</mark>, <small>small</small>, <abbr title="HyperText Markup Language">HTML</abbr></p>
<pre><code>&lt;p&gt;Example paragraph&lt;/p&gt;</code></pre>
</section>

<section>
<h2>Headings</h2>
<h1>Heading 1</h1>
<h2>Heading 2</h2>
<h3>Heading 3</h3>
<h4>Heading 4</h4>
<h5>Heading 5</h5>
<h6>Heading 6</h6>
</section>

<section>
<h2>Lists</h2>
<ul>
<li>Unordered item</li>
<li>Another item</li>
</ul>
<ol>
<li>Ordered item</li>
<li>Another ordered item</li>
</ol>
<dl>
<dt>Term</dt>
<dd>Definition</dd>
</dl>
</section>

<section>
<h2>Links & Media</h2>
<p><a href="#">A link</a></p>
<img src="https://via.placeholder.com/150" alt="Placeholder">
<video width="200" controls>
<source src="" type="video/mp4">
Your browser does not support the video tag.
</video>
</section>

<section>
<h2>Tables</h2>
<table border="1" cellpadding="5">
<tr><th>Header</th><th>Header</th></tr>
<tr><td>Cell</td><td>Cell</td></tr>
</table>
</section>

<section>
<h2>Form Elements</h2>
<form>
<label>Name: <input type="text" name="name"></label><br><br>
<label>Password: <input type="password" name="pass"></label><br><br>
<label>Checkbox: <input type="checkbox"></label><br><br>
<label>Radio: <input type="radio" name="r"></label><br><br>
<select>
<option>Option 1</option>
<option>Option 2</option>
</select><br><br>
<textarea rows="3" cols="30">Textarea</textarea><br><br>
<button type="submit">Submit</button>
</form>
</section>

<section>
<h2>Semantic HTML5 Elements</h2>
<header>Header element</header>
<nav>Navigation element</nav>
<main>Main content here</main>
<article>Article example</article>
<section>Section example</section>
<aside>Aside example</aside>
<footer>Footer example</footer>
</section>

<section>
<h2>Interactive Elements</h2>
<details>
<summary>Click to expand</summary>
Hidden details content.
</details>
</section>

<section>
<h2>Code & Preformatted</h2>
<pre><code>const x = 10;
console.log(x);</code></pre>
</section>

</body>
</html>
"""

path = Path("/mnt/data/html-elements-demo.html")
path.write_text(html_content, encoding="utf-8")
path
