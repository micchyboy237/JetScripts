import json
import os
import shutil
import sys
from typing import Literal
from pathlib import Path
from jet.file.utils import save_file
from jet.code.utils import remove_single_line_comments_preserving_triple_quotes
from jet.code.markdown_code_extractor import MarkdownCodeExtractor, CodeBlock


def extract_text_from_ipynb(notebook_path, include_outputs=True, include_code=False, include_comments=False, save_as: Literal['md', 'py'] = 'md'):
    """Extract text content from a Jupyter notebook and return as markdown or Python code."""
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)

        content = []
        text_blocks = []  # To collect consecutive markdown text

        for cell in notebook.get('cells', []):
            if cell['cell_type'] == 'markdown':
                if save_as == 'md':
                    content.extend(cell['source'])
                    content.append('')  # Newline separator
                else:  # save_as == 'py'
                    text_blocks.extend(cell['source'])
                    text_blocks.append('')  # Newline separator

            elif cell['cell_type'] == 'code':
                if text_blocks and save_as == 'py':
                    # Merge consecutive text blocks into a single triple-quoted string
                    content.append('"""')
                    content.extend(line.rstrip()
                                   for line in text_blocks if line.strip())
                    content.append('"""')
                    content.append('')
                    text_blocks = []  # Reset text blocks

                if include_code:
                    code_content = ''.join(cell['source'])
                    if not include_comments:
                        code_content = remove_single_line_comments_preserving_triple_quotes(
                            code_content)
                    if save_as == 'md':
                        content.append('```python')
                        content.append(code_content)
                        content.append('```')
                        content.append('')  # Newline separator
                    else:  # save_as == 'py'
                        content.append(code_content)
                        content.append('')  # Newline separator

                if include_outputs and save_as == 'md':
                    for output in cell.get('outputs', []):
                        if 'text' in output:
                            content.append('```output')
                            content.extend(output['text'])
                            content.append('```')
                            content.append('')
                        elif 'data' in output and 'text/plain' in output['data']:
                            content.append('```output')
                            content.extend(output['data']['text/plain'])
                            content.append('```')
                            content.append('')

        if text_blocks and save_as == 'py':
            # Append any remaining text blocks as a single triple-quoted string
            content.append('"""')
            content.extend(line.rstrip()
                           for line in text_blocks if line.strip())
            content.append('"""')
            content.append('')

        return '\n'.join(line.rstrip() for line in content)

    except Exception as e:
        print(f"Error processing {notebook_path}: {str(e)}")
        return None


def extract_text_from_md(md_path, include_code=True, include_comments=True, save_as: Literal['md', 'py'] = 'md'):
    """Extract text content from a markdown file using MarkdownCodeExtractor."""
    try:
        with open(md_path, 'r', encoding='utf-8') as f:
            content = f.read()

        extractor = MarkdownCodeExtractor()
        if save_as == 'md':
            if include_code:
                # Extract all content, including code blocks and text
                code_blocks = extractor.extract_code_blocks(
                    content, with_text=True)
                markdown_content = []
                for block in code_blocks:
                    if block['language'] == 'text':
                        markdown_content.append(block['code'])
                        markdown_content.append('')  # Newline separator
                    else:
                        if not include_comments and block['language'] == 'python':
                            code_content = remove_single_line_comments_preserving_triple_quotes(
                                block['code'])
                        else:
                            code_content = block['code']
                        markdown_content.append(f"```{block['language']}")
                        markdown_content.append(code_content)
                        markdown_content.append('```')
                        markdown_content.append('')  # Newline separator
                return '\n'.join(line.rstrip() for line in markdown_content)
            else:
                # Extract only non-code text, removing code blocks
                return extractor.remove_code_blocks(content, keep_file_paths=False)
        else:  # save_as == 'py'
            if include_code:
                code_blocks = extractor.extract_code_blocks(
                    content, with_text=True)
                python_content = []
                text_blocks = []
                for block in code_blocks:
                    if block['language'] == 'text':
                        text_blocks.append(block['code'])
                        text_blocks.append('')  # Newline separator
                    else:
                        if text_blocks:
                            # Merge consecutive text blocks into a single triple-quoted string
                            python_content.append('"""')
                            python_content.extend(
                                line.rstrip() for line in text_blocks if line.strip())
                            python_content.append('"""')
                            python_content.append('')
                            text_blocks = []
                        if block['language'] == 'python':
                            code_content = block['code']
                            if not include_comments:
                                code_content = remove_single_line_comments_preserving_triple_quotes(
                                    code_content)
                            python_content.append(code_content)
                            python_content.append('')  # Newline separator
                if text_blocks:
                    # Append any remaining text blocks
                    python_content.append('"""')
                    python_content.extend(line.rstrip()
                                          for line in text_blocks if line.strip())
                    python_content.append('"""')
                    python_content.append('')
                return '\n'.join(line.rstrip() for line in python_content)
            else:
                # Only extract text blocks as a single triple-quoted string
                code_blocks = extractor.extract_code_blocks(
                    content, with_text=True)
                text_blocks = [block['code']
                               for block in code_blocks if block['language'] == 'text']
                if text_blocks:
                    python_content = ['"""']
                    for text in text_blocks:
                        python_content.extend(
                            line.rstrip() for line in text.splitlines() if line.strip())
                        python_content.append('')  # Newline separator
                    python_content.append('"""')
                    python_content.append('')
                    return '\n'.join(line.rstrip() for line in python_content)
                return ''

    except Exception as e:
        print(f"Error processing {md_path}: {str(e)}")
        return None


def extract_text_from_py(py_path, include_code=True, include_comments=True, save_as: Literal['md', 'py'] = 'md'):
    """Extract text content from a Python file as markdown or Python code."""
    # Implement using ast
    pass


def process_file(input_path, output_dir=None, include_outputs=True, include_code=False, include_comments=False, save_as: Literal['md', 'py'] = 'md'):
    """Process a single file (notebook, markdown, or python) and save as markdown or Python code."""
    input_path = Path(input_path)

    if output_dir is None:
        output_dir = input_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{input_path.stem}.{save_as}"

    if input_path.suffix == '.ipynb':
        content = extract_text_from_ipynb(
            input_path,
            include_outputs=include_outputs,
            include_code=include_code,
            include_comments=include_comments,
            save_as=save_as
        )
    elif input_path.suffix == '.md':
        content = extract_text_from_md(
            input_path,
            include_code=include_code,
            include_comments=include_comments,
            save_as=save_as
        )
    elif input_path.suffix == '.py':
        content = extract_text_from_py(
            input_path,
            include_code=include_code,
            include_comments=include_comments,
            save_as=save_as
        )
    else:
        print(f"Unsupported file type: {input_path}")
        return

    if content:
        save_file(content, str(output_path))


def main():
    """Main function to process notebook, markdown, and python files."""
    input_path = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_python_modules/docs/hf_readmes"
    output_dir = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_python_modules/docs/hf_readmes/code"
    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)

    include_outputs = False
    include_code = True
    include_comments = True
    save_as: Literal['md', 'py'] = 'py'  # Change to 'md' for markdown output

    if os.path.isdir(input_path):
        for ext in ['*.ipynb', '*.md', '*.py']:
            for file in Path(input_path).rglob(ext):
                process_file(
                    file,
                    output_dir,
                    include_outputs=include_outputs,
                    include_code=include_code,
                    include_comments=include_comments,
                    save_as=save_as
                )
    else:
        process_file(
            input_path,
            output_dir,
            include_outputs=include_outputs,
            include_code=include_code,
            include_comments=include_comments,
            save_as=save_as
        )


if __name__ == "__main__":
    main()
