async def main():
    from IPython.display import display, HTML, Image
    from azure.ai.agents.models import CodeInterpreterTool
    from azure.ai.projects import AIProjectClient
    from azure.identity import DefaultAzureCredential
    from datetime import datetime
    from dotenv import load_dotenv
    from jet.logger import CustomLogger
    from pathlib import Path
    from typing import Any
    import os
    import shutil
    
    
    OUTPUT_DIR = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    LOG_DIR = f"{OUTPUT_DIR}/logs"
    
    log_file = os.path.join(LOG_DIR, "main.log")
    logger = CustomLogger(log_file, overwrite=True)
    logger.orange(f"Logs: {log_file}")
    
    
    
    load_dotenv()
    project_endpoint = os.environ["PROJECT_ENDPOINT"]
    
    project_client = AIProjectClient(
        endpoint=project_endpoint,
        credential=DefaultAzureCredential(),
    )
    
    
    
    async def run_agent_with_visualization():
        html_output = "<h2>Azure AI Agent Execution</h2>"
    
        with project_client:
            code_interpreter = CodeInterpreterTool()
    
            agent = project_client.agents.create_agent(
                model="llama3.2", log_dir=f"{LOG_DIR}/chats",
                name="my-agent",
                instructions="You are helpful agent",
                tools=code_interpreter.definitions,
            )
    
    
            html_output += f"<div><strong>Created agent</strong> with ID: {agent.id}</div>"
    
            thread = project_client.agents.threads.create()
            html_output += f"<div><strong>Created thread</strong> with ID: {thread.id}</div>"
    
            user_query = "Could you please create a bar chart for the operating profit using the following data and provide the file to me? Bali: 100 Travelers, Paris: 356 Travelers, London: 900 Travelers, Tokyo: 850 Travellers"
            html_output += "<div style='margin:15px 0; padding:10px; background-color:#f5f5f5; border-left:4px solid #007bff; border-radius:4px;'>"
            html_output += "<strong>User:</strong><br>"
            html_output += f"<div style='margin-left:15px'>{user_query}</div>"
            html_output += "</div>"
    
            message = project_client.agents.messages.create(
                thread_id=thread.id,
                role="user",
                content=user_query,
            )
    
            display(HTML(
                html_output + "<div style='color:#007bff'><i>Processing request...</i></div>"))
    
            run = project_client.agents.runs.create_and_process(
                thread_id=thread.id, agent_id=agent.id)
    
            status_color = 'green' if run.status == 'completed' else 'red'
            html_output += f"<div><strong>Run finished</strong> with status: <span style='color:{status_color}'>{run.status}</span></div>"
    
            if run.status == "failed":
                html_output += f"<div style='color:red'><strong>Run failed:</strong> {run.last_error}</div>"
    
            messages = project_client.agents.messages.list(thread_id=thread.id)
    
            html_output += "<div style='margin:15px 0; padding:10px; background-color:#f0f7ff; border-left:4px solid #28a745; border-radius:4px;'>"
            html_output += "<strong>Assistant:</strong><br>"
    
            try:
                assistant_msgs = [msg for msg in messages if hasattr(
                    msg, 'role') and msg.role == "assistant"]
    
                if assistant_msgs:
                    last_msg = assistant_msgs[-1]
                    if hasattr(last_msg, 'content'):
                        if isinstance(last_msg.content, list):
                            for content_item in last_msg.content:
                                if hasattr(content_item, 'type') and content_item.type == "text":
                                    html_output += f"<div style='margin-left:15px; white-space:pre-wrap'>{content_item.text.value}</div>"
                        elif isinstance(last_msg.content, str):
                            html_output += f"<div style='margin-left:15px; white-space:pre-wrap'>{last_msg.content}</div>"
    
                if not assistant_msgs:
                    if hasattr(messages, 'data'):
                        for msg in messages.data:
                            if hasattr(msg, 'role') and msg.role == "assistant":
                                if hasattr(msg, 'content'):
                                    html_output += f"<div style='margin-left:15px; white-space:pre-wrap'>{msg.content}</div>"
    
            except Exception as e:
                html_output += f"<div style='color:red'><strong>Error processing messages:</strong> {str(e)}</div>"
    
            html_output += "</div>"
    
            saved_images = []
            try:
                if hasattr(messages, 'image_contents'):
                    for image_content in messages.image_contents:
                        file_id = image_content.image_file.file_id
                        file_name = f"{file_id}_image_file.png"
                        project_client.agents.save_file(
                            file_id=file_id, file_name=file_name)
                        saved_images.append(file_name)
                        html_output += f"<div style='margin-top:10px'><strong>Generated Image:</strong> {file_name}</div>"
            except Exception as e:
                html_output += f"<div style='color:orange'><i>Note: No images found or error processing images</i></div>"
    
            try:
                if hasattr(messages, 'file_path_annotations'):
                    for file_path_annotation in messages.file_path_annotations:
                        file_name = Path(file_path_annotation.text).name
                        project_client.agents.save_file(
                            file_id=file_path_annotation.file_path.file_id, file_name=file_name)
                        html_output += "<div style='margin:10px 0; padding:8px; background-color:#f8f9fa; border:1px solid #ddd; border-radius:4px;'>"
                        html_output += f"<strong>Generated File:</strong> {file_name}<br>"
                        html_output += f"<strong>Type:</strong> {file_path_annotation.type}<br>"
                        html_output += "</div>"
            except Exception as e:
                html_output += f"<div style='color:orange'><i>Note: No file annotations found or error processing files</i></div>"
    
            project_client.agents.delete_agent(agent.id)
            html_output += "<div style='margin-top:10px'><i>Agent deleted after completion</i></div>"
    
            display(HTML(html_output))
    
            for img_file in saved_images:
                display(Image(img_file))
    
    await run_agent_with_visualization()
    
    logger.info("\n\n[DONE]", bright=True)

if __name__ == '__main__':
    import asyncio
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(main())
        else:
            loop.run_until_complete(main())
    except RuntimeError:
        asyncio.run(main())