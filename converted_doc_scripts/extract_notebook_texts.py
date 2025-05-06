import json
import os
import sys
from pathlib import Path


def extract_text_from_ipynb(notebook_path):
    """Extract text content from a Jupyter notebook and return as markdown."""
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)

        markdown_content = []

        for cell in notebook.get('cells', []):
            if cell['cell_type'] == 'markdown':
                # Add markdown cell content directly
                markdown_content.extend(cell['source'])
            elif cell['cell_type'] == 'code':
                # Add code cell content wrapped in triple backticks
                markdown_content.append('```python')
                markdown_content.extend(cell['source'])
                markdown_content.append('```')
                # Add code cell outputs if they exist
                for output in cell.get('outputs', []):
                    if 'text' in output:
                        markdown_content.append('```output')
                        markdown_content.extend(output['text'])
                        markdown_content.append('```')
                    elif 'data' in output and 'text/plain' in output['data']:
                        markdown_content.append('```output')
                        markdown_content.extend(output['data']['text/plain'])
                        markdown_content.append('```')

        # Join lines and ensure proper newline handling
        return '\n'.join(line.rstrip() for line in markdown_content)

    except Exception as e:
        print(f"Error processing {notebook_path}: {str(e)}")
        return None


def save_to_markdown(content, output_path):
    """Save content to a markdown file."""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Successfully saved markdown to {output_path}")
    except Exception as e:
        print(f"Error saving markdown to {output_path}: {str(e)}")


def process_notebook(input_path, output_dir=None):
    """Process a single notebook file and save as markdown."""
    input_path = Path(input_path)

    # Determine output path
    if output_dir is None:
        output_dir = input_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{input_path.stem}.md"

    # Extract content and save
    content = extract_text_from_ipynb(input_path)
    if content:
        save_to_markdown(content, output_path)


def main():
    """Main function to process notebook files."""
    if len(sys.argv) < 2:
        print(
            "Usage: python ipynb_to_markdown.py <notebook_path> [output_dir]")
        sys.exit(1)

    notebook_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None

    # Check if input is a directory or single file
    if os.path.isdir(notebook_path):
        for file in Path(notebook_path).glob('*.ipynb'):
            process_notebook(file, output_dir)
    else:
        process_notebook(notebook_path, output_dir)


if __name__ == "__main__":
    main()
