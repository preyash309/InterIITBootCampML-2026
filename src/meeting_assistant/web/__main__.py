"""Single-process demo server; no reload or multiple GPU-owning workers."""

import logging
import sys

import uvicorn

from .app import create_app
from .config import WebConfig


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    config = WebConfig.from_env()
    # Range seeking cancels old HTTP connections. Selector avoids noisy Windows
    # Proactor pipe-reset callbacks; all model/FFmpeg work runs in the worker thread.
    loop = "asyncio:SelectorEventLoop" if sys.platform == "win32" else "auto"
    uvicorn.run(create_app(config), host=config.host, port=config.port, workers=1, loop=loop)


if __name__ == "__main__":
    main()
