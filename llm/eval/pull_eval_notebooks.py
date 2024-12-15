import os
import codecs
import json

# Define input directory
input_dir = "/Users/jethroestrada/Desktop/External_Projects/AI/libs/llama_index/docs/docs/examples/evaluation"
generated_dir = "notebooks"

# Read .ipynb files
files = [os.path.join(input_dir, f)
         for f in os.listdir(input_dir) if f.endswith(".ipynb")]
print(f"Found {len(files)} .ipynb files: {files}")

# Function to extract Python code cells from a file


def read_file(file):
    with codecs.open(file, 'r', encoding='utf-8') as f:
        source = f.read()

    source_dict = json.loads(source)
    cells = source_dict.get('cells', [])
    source_lines = []

    for cell in cells:
        code_lines = []
        for line in cell.get('source', []):
            if cell.get('cell_type') == 'code':
                # Remove commented lines
                if line.strip().startswith('#'):
                    continue
                # Comment out installation lines
                elif not line.strip().startswith('%') and not line.strip().startswith('!'):
                    if not line.endswith('\n'):
                        line += '\n'

                code_lines.append(line)
            # else:
            #     if not line.strip().startswith('#'):
            #         line = "# " + line
            #     if not line.endswith('\n'):
            #         line += '\n'
            # code_lines.append(line)
        source_lines.extend(code_lines)

    return "".join(source_lines)


# Process each file and extract Python code
for file in files:
    file_path = os.path.join(input_dir, file)
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    print(f"Processing file: {file_name}...")

    try:
        content = read_file(file_path)

        output_dir = os.path.join(os.path.dirname(__file__), generated_dir)
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{file_name}.py")

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(content)

    except Exception as e:
        print(f"Failed to process file {file_name}: {e}")

print(f"Total files processed: {len(files)}")
