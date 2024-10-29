import subprocess
from pathlib import Path

import ffmpeg
import Flask

app = Flask(__name__)


@app.route("/init")
def init():
    return "init"
