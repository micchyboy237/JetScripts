import justext

from jet.file.utils import load_file
from jet.utils.commands import copy_to_clipboard

html_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/top_isekai_anime_2025/pages/animebytes_in_15_best_upcoming_isekai_anime_in_2025/page.html"

html_content = load_file(html_file)

# Process HTML with jusText
paragraphs = justext.justext(html_content, justext.get_stoplist("English"))

# Extract non-boilerplate text
clean_text = []
for paragraph in paragraphs:
    if not paragraph.is_boilerplate:
        clean_text.append(paragraph.text)

# Join paragraphs into a single string
result = "\n".join(clean_text)
print(result)
copy_to_clipboard(result)
