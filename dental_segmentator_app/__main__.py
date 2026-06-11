import os
from pathlib import Path

from watchfiles import run_process

from .gradio_app import launch


if __name__ == "__main__":
    if os.environ.get("HOT_RELOAD", "0") == "1":
        app_dir = Path(__file__).resolve().parent
        run_process(str(app_dir), target=launch)
    else:
        launch()
