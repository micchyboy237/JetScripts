import asyncio
from typing import List, Optional
import uuid
import json
import subprocess
from jet.llm.mlx.mlx_types import ModelType
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from jet.executor.command import run_command
from jet.logger import logger
import shlex
import re

app = FastAPI(title="Parallel MLX Stream Generation Server")

# Add CORS middleware to allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["Content-Type", "Accept"],
)

semaphore = asyncio.Semaphore(4)
MPIRUN_GENERATE_FILE = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/test/_test_for_running_temp_scripts.py"


class GenerateRequest(BaseModel):
    model: ModelType
    prompts: List[str]
    max_tokens: int = 100
    temp: float = 0.7
    verbose: bool = False
    task_id: Optional[str] = None


class StreamingError(Exception):
    """Custom exception for handling streaming-specific errors."""
    pass


async def run_mpi_process(
    prompts: List[str],
    model: ModelType = "llama-3.2-1b-instruct-4bit",
    max_tokens: Optional[int] = None,
    temp: Optional[float] = None,
    verbose: Optional[bool] = None,
    task_id: Optional[str] = None
):
    """Run the MPI script as a subprocess and stream its output."""
    task_id = task_id or str(uuid.uuid4())
    input_data = {
        "model": model,
        "prompts": prompts,
        "max_tokens": max_tokens,
        "temp": temp,
        "verbose": verbose,
        "task_id": task_id
    }
    input_json = json.dumps(input_data)
    command = f"mpirun -np 4 python {MPIRUN_GENERATE_FILE} {shlex.quote(input_json)}"

    async def stream_output():
        error_messages = []
        try:
            for line in run_command(command, separator=" "):
                line = line.strip()
                if line.startswith("error: "):
                    error_messages.append(line[7:].strip())
                    yield json.dumps({"type": "error", "message": line[7:].strip()}) + "\n"
                elif line.startswith("data: "):
                    content = line[6:].strip()
                    # Parse data lines for token updates
                    token_match = re.match(
                        r"Process \d+ \(Prompt ID: ([^)]+)\): (.+)", content)
                    if token_match:
                        prompt_id, token = token_match.groups()
                        # Find the prompt associated with this prompt_id from input_data
                        prompt = next((p for i, p in enumerate(
                            prompts) if i % 4 == int(content.split()[1])), prompts[0])
                        yield json.dumps({
                            "type": "token",
                            "prompt_id": prompt_id,
                            "task_id": task_id,
                            "prompt": prompt,
                            "token": token
                        }) + "\n"
                    # Parse result lines
                    result_match = re.match(r"Result \d+: (.+)", content)
                    if result_match:
                        result_json = json.loads(result_match.group(1))
                        yield json.dumps({
                            "type": "result",
                            "prompt": result_json["prompt"],
                            "response": result_json["response"],
                            "prompt_id": result_json["prompt_id"],
                            "task_id": result_json["task_id"]
                        }) + "\n"
                elif line.startswith("result: "):
                    # Parse result lines directly
                    result_json = json.loads(
                        line[8:].replace("Result ", "")[2:])
                    yield json.dumps({
                        "type": "result",
                        "prompt": result_json["prompt"],
                        "response": result_json["response"],
                        "prompt_id": result_json["prompt_id"],
                        "task_id": result_json["task_id"]
                    }) + "\n"
            if error_messages:
                raise StreamingError(
                    f"Streaming errors: {'\n'.join(error_messages)}")
        except StreamingError as e:
            error_message = str(e)
            logger.error(f"Streaming failed:\n{error_message}")
            yield json.dumps({"type": "error", "message": error_message}) + "\n"
            raise
        except Exception as e:
            error_message = str(e)
            unique_errors = [
                err for err in error_messages if err not in error_message]
            if unique_errors:
                error_message += f"; Other errors encountered: {'; '.join(unique_errors)}"
            logger.error(f"Unexpected error during streaming: {error_message}")
            yield json.dumps({"type": "error", "message": error_message}) + "\n"
            raise RuntimeError(error_message)
        finally:
            pass

    return stream_output


@app.post("/generate")
async def generate(request: GenerateRequest):
    """Endpoint to generate text for given prompts in parallel."""
    async with semaphore:
        try:
            stream_gen = await run_mpi_process(
                model=request.model,
                prompts=request.prompts,
                max_tokens=request.max_tokens,
                temp=request.temp,
                verbose=request.verbose,
                task_id=request.task_id
            )
            return StreamingResponse(
                stream_gen(),
                media_type="application/json"
            )
        except StreamingError as e:
            raise HTTPException(
                status_code=500, detail=f"Streaming error: {str(e)}")
        except RuntimeError as e:
            raise HTTPException(status_code=500, detail=str(e))
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"Unexpected error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("_temp_test2:app", host="0.0.0.0", port=9000, reload=True)
