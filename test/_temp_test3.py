from mrkdwn_analysis import MarkdownAnalyzer

analyzer = MarkdownAnalyzer("path/to/document.md")

headers = analyzer.identify_headers()
paragraphs = analyzer.identify_paragraphs()
links = analyzer.identify_links()
...
