import asyncio
from jet.transformers.formatters import format_json
from autogen_core import CancellationToken
from autogen_core.code_executor import CodeBlock
from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
from jet.logger import CustomLogger
from pathlib import Path
import os
import shutil
import venv


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Command Line Code Executors

Command line code execution is the simplest form of code execution.
Generally speaking, it will save each code block to a file and then execute that file.
This means that each code block is executed in a new process. There are two forms of this executor:

- Docker ({py:class}`~autogen_ext.code_executors.docker.DockerCommandLineCodeExecutor`) - this is where all commands are executed in a Docker container
- Local ({py:class}`~autogen_ext.code_executors.local.LocalCommandLineCodeExecutor`) - this is where all commands are executed on the host machine

## Docker

```{note}
To use {py:class}`~autogen_ext.code_executors.docker.DockerCommandLineCodeExecutor`, ensure the `autogen-ext[docker]` package is installed. For more details, see the [Packages Documentation](https://microsoft.github.io/autogen/dev/packages/index.html).

```

The {py:class}`~autogen_ext.code_executors.docker.DockerCommandLineCodeExecutor` will create a Docker container and run all commands within that container. 
The default image that is used is `python:3-slim`, this can be customized by passing the `image` parameter to the constructor. 
If the image is not found locally then the class will try to pull it. 
Therefore, having built the image locally is enough. The only thing required for this image to be compatible with the executor is to have `sh` and `python` installed. 
Therefore, creating a custom image is a simple and effective way to ensure required system dependencies are available.

You can use the executor as a context manager to ensure the container is cleaned up after use. 
Otherwise, the `atexit` module will be used to stop the container when the program exits.

### Inspecting the container

If you wish to keep the container around after AutoGen is finished using it for whatever reason (e.g. to inspect the container), 
then you can set the `auto_remove` parameter to `False` when creating the executor. 
`stop_container` can also be set to `False` to prevent the container from being stopped at the end of the execution.

### Example
"""
logger.info("# Command Line Code Executors")



work_dir = Path("coding")
work_dir.mkdir(exist_ok=True)

async def async_func_9():
    async with DockerCommandLineCodeExecutor(work_dir=work_dir) as executor:  # type: ignore
        logger.debug(
            await executor.execute_code_blocks(
                code_blocks=[
                    CodeBlock(language="python", code="logger.debug('Hello, World!')"),
                ],
                cancellation_token=CancellationToken(),
            )
        )
    return result

result = asyncio.run(async_func_9())
logger.success(format_json(result))

"""
### Combining an Application in Docker with a Docker based executor

It is desirable to bundle your application into a Docker image. But then, how do you allow your containerised application to execute code in a different container?

The recommended approach to this is called "Docker out of Docker", where the Docker socket is mounted to the main AutoGen container, so that it can spawn and control "sibling" containers on the host. This is better than what is called "Docker in Docker", where the main container runs a Docker daemon and spawns containers within itself. You can read more about this [here](https://jpetazzo.github.io/2015/09/03/do-not-use-docker-in-docker-for-ci/).

To do this you would need to mount the Docker socket into the container running your application. This can be done by adding the following to the `docker run` command:

```bash
-v /var/run/docker.sock:/var/run/docker.sock
```

This will allow your application's container to spawn and control sibling containers on the host.

If you need to bind a working directory to the application's container but the directory belongs to your host machine, 
use the `bind_dir` parameter. This will allow the application's container to bind the *host* directory to the new spawned containers and allow it to access the files within the said directory. If the `bind_dir` is not specified, it will fallback to `work_dir`.

## Local

```{attention}
The local version will run code on your local system. Use it with caution.
```

To execute code on the host machine, as in the machine running your application, {py:class}`~autogen_ext.code_executors.local.LocalCommandLineCodeExecutor` can be used.

### Example
"""
logger.info("### Combining an Application in Docker with a Docker based executor")



work_dir = Path("coding")
work_dir.mkdir(exist_ok=True)

local_executor = LocalCommandLineCodeExecutor(work_dir=work_dir)
logger.debug(
    await local_executor.execute_code_blocks(
        code_blocks=[
            CodeBlock(language="python", code="logger.debug('Hello, World!')"),
        ],
        cancellation_token=CancellationToken(),
    )
)

"""
## Local within a Virtual Environment

If you want the code to run within a virtual environment created as part of the application’s setup, you can specify a directory for the newly created environment and pass its context to  {py:class}`~autogen_ext.code_executors.local.LocalCommandLineCodeExecutor`. This setup allows the executor to use the specified virtual environment consistently throughout the application's lifetime, ensuring isolated dependencies and a controlled runtime environment.
"""
logger.info("## Local within a Virtual Environment")



work_dir = Path("coding")
work_dir.mkdir(exist_ok=True)

venv_dir = work_dir / ".venv"
venv_builder = venv.EnvBuilder(with_pip=True)
venv_builder.create(venv_dir)
venv_context = venv_builder.ensure_directories(venv_dir)

local_executor = LocalCommandLineCodeExecutor(work_dir=work_dir, virtual_env_context=venv_context)
await local_executor.execute_code_blocks(
    code_blocks=[
        CodeBlock(language="bash", code="pip install matplotlib"),
    ],
    cancellation_token=CancellationToken(),
)

"""
As we can see, the code has executed successfully, and the installation has been isolated to the newly created virtual environment, without affecting our global environment.
"""
logger.info("As we can see, the code has executed successfully, and the installation has been isolated to the newly created virtual environment, without affecting our global environment.")

logger.info("\n\n[DONE]", bright=True)