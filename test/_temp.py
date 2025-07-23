import os
import shutil
from jet.code.markdown_utils._markdown_parser import parse_markdown
from jet.file.utils import load_file, save_file


md_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/data/complete_jet_resume.md"

md_content = load_file(md_file)

results_ignore_links = parse_markdown(
    md_content, ignore_links=True, merge_contents=False, merge_headers=False)
results_with_links = parse_markdown(
    md_content, ignore_links=False, merge_contents=False, merge_headers=False)

output_dir = os.path.join(os.path.dirname(
    __file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(output_dir, ignore_errors=True)

save_file(results_ignore_links, f"{output_dir}/results_ignore_links.json")
save_file(results_with_links, f"{output_dir}/results_with_links.json")
