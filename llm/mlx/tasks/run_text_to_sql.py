from jet.llm.mlx.tasks.text_to_sql import TextToSQLResult, text_to_sql
from jet.models.model_types import LLMModelType
from jet.logger import logger

if __name__ == "__main__":
    model: LLMModelType = "llama-3.2-3b-instruct-4bit"
    input_text: str = "Find all employees who work in the Sales department and have a salary greater than 50000."
    schema: str = (
        "Table: Employees\n"
        "Columns:\n"
        "- employee_id (INT, PRIMARY KEY)\n"
        "- first_name (VARCHAR)\n"
        "- last_name (VARCHAR)\n"
        "- department (VARCHAR)\n"
        "- salary (DECIMAL)"
    )

    for method in ["stream_generate", "generate_step"]:
        logger.log("Method:", method, colors=["GRAY", "WHITE"])
        result: TextToSQLResult = text_to_sql(
            input_text, schema, model, method=method
        )
        logger.log("Input Text:", input_text, colors=["GRAY", "DEBUG"])
        logger.log("Schema:", schema, colors=["GRAY", "DEBUG"])
        logger.log("SQL Query:", result["sql_query"], colors=[
                   "GRAY", "SUCCESS"])
        logger.log("Valid:", result["is_valid"], colors=[
                   "GRAY", "SUCCESS" if result["is_valid"] else "ERROR"])
        if result["error"]:
            logger.log("Error:", result["error"], colors=["GRAY", "ERROR"])
        else:
            logger.log("Output is valid.", method, colors=["GRAY", "SUCCESS"])
        logger.newline()
