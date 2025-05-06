import json
import os
import shutil
import sys
from pathlib import Path

from jet.file.utils import save_file


def extract_text_from_ipynb(notebook_path, include_outputs=True, include_code=False):
    """Extract text content from a Jupyter notebook and return as markdown."""
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)

        markdown_content = []

        for cell in notebook.get('cells', []):
            if cell['cell_type'] == 'markdown':
                markdown_content.extend(cell['source'])
                markdown_content.append('')  # Newline separator

            elif cell['cell_type'] == 'code':
                if include_code:
                    markdown_content.append('```python')
                    markdown_content.extend(cell['source'])
                    markdown_content.append('```')
                    markdown_content.append('')  # Newline separator

                if include_outputs:
                    for output in cell.get('outputs', []):
                        if 'text' in output:
                            markdown_content.append('```output')
                            markdown_content.extend(output['text'])
                            markdown_content.append('```')
                            markdown_content.append('')
                        elif 'data' in output and 'text/plain' in output['data']:
                            markdown_content.append('```output')
                            markdown_content.extend(
                                output['data']['text/plain'])
                            markdown_content.append('```')
                            markdown_content.append('')

        return '\n'.join(line.rstrip() for line in markdown_content)

    except Exception as e:
        print(f"Error processing {notebook_path}: {str(e)}")
        return None


def process_notebook(input_path, output_dir=None, include_outputs=True, include_code=False):
    """Process a single notebook file and save as markdown."""
    input_path = Path(input_path)

    if output_dir is None:
        output_dir = input_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{input_path.stem}.md"

    content = extract_text_from_ipynb(
        input_path,
        include_outputs=include_outputs,
        include_code=include_code
    )
    if content:
        save_file(content, str(output_path))


def main():
    """Main function to process notebook files."""
    notebook_path = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/converted_doc_scripts/all-rag-techniques/notebooks"
    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)

    include_outputs = False
    include_code = False

    if os.path.isdir(notebook_path):
        for file in Path(notebook_path).glob('*.ipynb'):
            process_notebook(
                file, output_dir, include_outputs=include_outputs, include_code=include_code)
    else:
        process_notebook(notebook_path, output_dir,
                         include_outputs=include_outputs, include_code=include_code)


if __name__ == "__main__":
    main()
