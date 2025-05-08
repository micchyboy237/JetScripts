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
import os

# In _temp_test2.py, before running the command
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

app = FastAPI(title="Parallel MLX Stream Generation Server")
semaphore = asyncio.Semaphore(4)
MPIRUN_GENERATE_FILE = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/test/_test_for_running_temp_scripts.py"


class GenerateRequest(BaseModel):
    model: ModelType
    prompts: List[str]
    max_tokens: int = 100
    temp: float = 0.7
    verbose: bool = False


class StreamingError(Exception):
    """Custom exception for handling streaming-specific errors."""
    pass


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
                # else:
                #     yield f"other: {line.strip()}\n\n"
            if error_messages:
                raise StreamingError(
                    f"Streaming errors: {'\n'.join(error_messages)}")
        except StreamingError as e:
            error_message = str(e)
            logger.error(f"Streaming failed:\n{error_message}")
            yield f"error: {error_message}\n\n"
            raise
        except Exception as e:
            error_message = str(e)
            # Only append error_messages that are not duplicates of the current exception
            unique_errors = [
                err for err in error_messages if err not in error_message]
            if unique_errors:
                error_message += f"; Other errors encountered: {'; '.join(unique_errors)}"
            logger.error(f"Unexpected error during streaming: {error_message}")
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
