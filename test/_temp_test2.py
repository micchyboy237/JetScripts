import os

import numpy as np
from smolagents import LocalPythonExecutor
from smolagents.local_python_executor import InterpreterError

# ============================
# 1. Create the executor
# ============================
executor = LocalPythonExecutor(
    additional_authorized_imports=[
        "numpy",
        "pandas",
        "matplotlib",
        "matplotlib.pyplot",
    ],
    timeout_seconds=30,
    max_print_outputs_length=100_000,
)

# Send base tools → this injects print + final_answer tool
executor.send_tools({})


# ============================
# 2. Add custom tool
# ============================
def calculate_statistics(data: list[float]) -> dict:
    """Calculate basic statistics for a list of numbers."""
    arr = np.array(data)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


executor.send_tools({"calculate_statistics": calculate_statistics})

# ============================
# 3. Send initial variables
# ============================
executor.send_variables(
    {
        "initial_data": [23, 45, 67, 12, 89, 34, 56, 78],
        "company": "TechCorp",
    }
)

# ============================
# 4. Consolidated code block
# ============================
full_code = """
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print(f"Company: {company}")
print(f"Initial data: {initial_data}")

# Create DataFrame
df = pd.DataFrame({
    "values": initial_data,
    "category": ["A", "B", "A", "C", "B", "A", "C", "B"]
})
print("DataFrame created:")
print(df)

# Statistics
stats = calculate_statistics(initial_data)
print("Statistics:", stats)

# Update DataFrame
df["doubled"] = df["values"] * 2
print("Updated DataFrame head:")
print(df.head())

# Plotting (headless-safe)
plt.switch_backend('Agg')

plt.figure(figsize=(8, 5))
plt.bar(df["category"], df["values"], color="skyblue")
plt.title(f"Values by Category - {company}")
plt.xlabel("Category")
plt.ylabel("Value")
plt.grid(axis="y", alpha=0.3)
plt.savefig("output_plot.png")
print("Plot successfully saved as output_plot.png")

# === FINAL ANSWER (Correct way for direct executor) ===
final_answer = {
    "statistics": stats,
    "dataframe_shape": df.shape,
    "mean_value": stats["mean"],
    "plot_file": "output_plot.png",
    "message": "Execution completed successfully with plot saved."
}

print("Final answer prepared.")
"""

# ============================
# 5. Execute
# ============================
print("=== Starting Execution ===\n")

try:
    result = executor(full_code)

    print("Output (returned value):", result.output)
    if result.logs.strip():
        print("\n=== Logs ===\n", result.logs.strip())

    print(f"\nIs final answer detected: {result.is_final_answer}")

    # The actual final result is usually in result.output when using final_answer = ...
    if result.output is not None:
        print("\n=== Final Structured Answer ===")
        print(result.output)

except InterpreterError as e:
    print(f"InterpreterError: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")

print("\n=== Execution Finished ===")

# Check plot file
if os.path.exists("output_plot.png"):
    print("Plot file created successfully at: output_plot.png")
else:
    print("Plot file was not created.")
