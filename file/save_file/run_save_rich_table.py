import os
import shutil
from rich.table import Table
from jet.file.utils import save_file

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

def main():
    # Define output file paths
    file_path1 = os.path.join(OUTPUT_DIR, "results1.json")
    file_path2 = os.path.join(OUTPUT_DIR, "results2.json")
    table_path_json = os.path.join(OUTPUT_DIR, "table_results.json")
    table_path_jsonl = os.path.join(OUTPUT_DIR, "table_results.jsonl")
    table_path_md = os.path.join(OUTPUT_DIR, "table_results.md")

    # Existing sample results
    sample_results_1 = {
        "session_data": [
            "item1",
            2,
            {
                "item3": {
                    "task1": {"status": "completed", "details": {"time": "2 hours"}},
                    "task2": {"status": "pending", "priority": "high"},
                },
            }
        ]
    }
    sample_results_2 = [
        {
            "user_data": {
                "id": 1,
                "name": "John Doe",
                "preferences": {
                    "theme": True,
                    "notifications": "dark"
                }
            },
        }
    ]
    save_file(sample_results_1, file_path1)
    save_file(sample_results_2, file_path2)

    # Example 1: Simple Table with user data saved as JSON
    user_table = Table(title="User Profiles")
    user_table.add_column("ID", style="cyan")
    user_table.add_column("Name", style="magenta")
    user_table.add_column("Role", style="green")
    user_table.add_row("1", "Alice Smith", "Developer")
    user_table.add_row("2", "Bob Jones", "Designer")
    save_file(user_table, table_path_json, verbose=True)

    # Example 2: Table with nested data saved as JSONL (append mode)
    project_table = Table(title="Project Status")
    project_table.add_column("Project ID", style="cyan")
    project_table.add_column("Name", style="magenta")
    project_table.add_column("Status", style="green")
    project_table.add_column("Details", style="yellow")
    project_table.add_row(
        "P001",
        "Website Redesign",
        "In Progress",
        '{"progress": 75, "deadline": "2025-12-01"}'
    )
    project_table.add_row(
        "P002",
        "API Development",
        "Completed",
        '{"progress": 100, "deadline": "2025-10-15"}'
    )
    save_file(project_table, table_path_jsonl, verbose=True, append=True)

    # Example 3: Table with numeric data saved as Markdown
    sales_table = Table(title="Monthly Sales")
    sales_table.add_column("Month", style="cyan")
    sales_table.add_column("Revenue", style="magenta")
    sales_table.add_column("Units Sold", style="green")
    sales_table.add_row("Jan", "$10,000", "150")
    sales_table.add_row("Feb", "$12,500", "180")
    sales_table.add_row("Mar", "$15,000", "200")
    save_file(sales_table, table_path_md, verbose=True)

if __name__ == "__main__":
    main()
