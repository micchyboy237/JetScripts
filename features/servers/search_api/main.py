import os
import uvicorn
from jet.logger import logger
import sys
import traceback


def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    log_level: str = "info",
    reload: bool = False,
    reload_dirs: list[str] | str | None = None
) -> None:
    """
    Run the FastAPI server with specified configuration.

    Args:
        host: Host address to bind the server (default: "0.0.0.0")
        port: Port number to run the server (default: 8000)
        log_level: Logging level for uvicorn (default: "info")
        reload: Enable auto-reload for development (default: False)
    """
    logger.debug(
        f"Initializing server startup with host={host}, port={port}, log_level={log_level}, reload={reload}")
    try:
        logger.info(
            f"Starting FastAPI server on {host}:{port} with log_level={log_level}, reload={reload}")
        uvicorn.run(
            "server.search_api:app",  # Use import string to support reload
            host=host,
            port=port,
            log_level=log_level,
            reload=reload,
            reload_dirs=reload_dirs,
        )
        logger.info("Server shutdown gracefully")
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
        logger.debug(f"Stack trace: {traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    logger.debug("Starting server from __main__")
    base_dir = os.path.dirname(__file__)
    run_server(reload=True, reload_dirs=[
        f"{base_dir}/main.py",
        f"{base_dir}/server",
    ])
