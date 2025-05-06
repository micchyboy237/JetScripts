import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import subprocess
import json
import uuid
from typing import List

app = FastAPI(title="Parallel MLX Stream Generation Server")

# Semaphore to limit concurrent tasks to 4
semaphore = asyncio.Semaphore(4)


class GenerateRequest(BaseModel):
    prompts: List[str]
    max_tokens: int = 100
    temp: float = 0.7
    verbose: bool = False


async def run_mpi_process(prompts: List[str], max_tokens: int, temp: float, verbose: bool):
    """Run the MPI script as a subprocess and stream its output."""
    task_id = str(uuid.uuid4())
    input_data = {
        "prompts": prompts,
        "max_tokens": max_tokens,
        "temp": temp,
        "verbose": verbose,
        "task_id": task_id
    }
    with open(f"task_{task_id}.json", "w") as f:
        json.dump(input_data, f)
    process = await asyncio.create_subprocess_exec(
        "mpirun", "-np", "4", "python", "mpi_generate.py", f"task_{task_id}.json",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )

    async def stream_output():
        while True:
            line = await process.stdout.readline()
            if not line:
                break
            yield line.decode().strip() + "\n"
        await process.wait()
        import os
        os.remove(f"task_{task_id}.json")
    return stream_output()


@app.post("/generate")
async def generate(request: GenerateRequest):
    """Endpoint to generate text for given prompts in parallel."""
    async with semaphore:
        try:
            return StreamingResponse(
                run_mpi_process(
                    prompts=request.prompts,
                    max_tokens=request.max_tokens,
                    temp=request.temp,
                    verbose=request.verbose
                ),
                media_type="text/plain"
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
