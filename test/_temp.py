from jet.transformers.formatters import format_json
import stanza
import markdown
import html2text  # Added for plain text conversion
from jet.logger import logger

# Initialize Stanza pipeline
nlp = stanza.Pipeline('en', processors='tokenize,pos', verbose=True)

# Markdown text input
markdown_text = """
## Section
First sentence ends here. Second with noun.

Bullet: Action item.
Next para.
"""

# Convert Markdown to HTML, then to plain text
html = markdown.markdown(markdown_text, output_format='html')
h = html2text.HTML2Text()
h.body_only = True  # Exclude HTML wrapper tags
plain_text = h.handle(html).strip()

# Process with Stanza
doc = nlp(plain_text)
sentences = [sent.text.strip() for sent in doc.sentences]
paragraphs = [para.strip() for para in plain_text.split('\n\n') if para.strip()]  # Refined split

# Output results
logger.gray("Sentences:")
logger.success(format_json(sentences))
logger.gray("Paragraphs:")
logger.success(format_json(paragraphs))

# Optional POS inspection
for sent in doc.sentences:
    for word in sent.words:
        print(word.text, word.xpos)