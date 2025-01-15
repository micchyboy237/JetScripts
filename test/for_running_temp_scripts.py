import os
from jet.logger import logger
from jet.transformers.formatters import format_json
from llama_index.core.readers.file.base import SimpleDirectoryReader

if __name__ == "__main__":
    data_path = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data/companies.md"
    source_data = {
        "lines": [
            14,
        ],
        "question": "Can you describe your experience developing a social networking app (Graduapp) for students, parents, teachers, and schools at 8WeekApp?",
        "answer": "Yes, I developed a social networking app (Graduapp) for students, parents, teachers, and schools. The app serves as an online journal of their experience as a student at their institution. I utilized key technologies such as React, React Native, Node.js, Firebase, and MongoDB.",
        "sources": [
            "- Key technologies: React, React Native, Node.js, Firebase, MongoDB"
        ]
    }
    with open(data_path) as file:
        # Read the entire content of the file
        file_content = file.read()
    file_content_lines = file_content.splitlines()
    source_line_indexes = [line - 1 for line in source_data["lines"]]
    start_index = source_line_indexes[0]
    end_index = source_line_indexes[-1] if len(
        source_line_indexes) > 1 else start_index + 1
    source_lines = file_content_lines[start_index:end_index]
    source_lines = [line for line in source_lines if line.strip()]

    logger.debug(f"Results ({len(source_lines)}):")
    logger.success(format_json(source_lines))
