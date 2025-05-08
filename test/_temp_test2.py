import asyncio
from typing import List, Optional
import uuid
import json
import subprocess
from jet.llm.mlx.mlx_types import ModelType
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from fastapi import FastAPI, HTTPException
from jet.executor.command import run_command
from jet.logger import logger
import shlex

app = FastAPI(title="Parallel MLX Stream Generation Server")
semaphore = asyncio.Semaphore(4)
MPIRUN_GENERATE_FILE = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/test/_test_for_running_temp_scripts.py"


class GenerateRequest(BaseModel):
    model: ModelType
    prompts: List[str]
    max_tokens: int = 100
    temp: float = 0.7
    verbose: bool = False


async def run_mpi_process(
    prompts: List[str],
    model: ModelType = "llama-3.2-1b-instruct-4bit",
    max_tokens: Optional[int] = None,
    temp: Optional[float] = None,
    verbose: Optional[bool] = None
):
    """Run the MPI script as a subprocess and stream its output."""
    task_id = str(uuid.uuid4())
    input_data = {
        "model": model,
        "prompts": prompts,
        "max_tokens": max_tokens,
        "temp": temp,
        "verbose": verbose,
        "task_id": task_id
    }
    input_json = json.dumps(input_data)
    # Use shlex.quote to properly escape the JSON string
    command = f"mpirun -np 4 python {MPIRUN_GENERATE_FILE} {shlex.quote(input_json)}"

    async def stream_output():
        error_messages = []
        try:
            for line in run_command(command, separator=" "):
                if line.startswith("error: "):
                    error_messages.append(line[7:].strip())
                    yield line
                elif line.startswith(("result: ", "data: ")):
                    yield line
                else:
                    yield f"data: {line.strip()}\n\n"
            if error_messages:
                logger.error(
                    f"Errors encountered during streaming: {error_messages}")
                raise RuntimeError(
                    f"Streaming errors: {'; '.join(error_messages)}")
        except Exception as e:
            error_message = str(e)
            if error_messages:
                error_message += f"; Additional errors: {'; '.join(error_messages)}"
            logger.error(f"Streaming failed: {error_message}")
            yield f"error: {error_message}\n\n"
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
                verbose=request.verbose
            )
            return StreamingResponse(
                stream_gen(),
                media_type="text/event-stream"
            )
        except RuntimeError as e:
            raise HTTPException(status_code=500, detail=str(e))
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"Unexpected error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("_temp_test2:app", host="0.0.0.0", port=9000, reload=True)
