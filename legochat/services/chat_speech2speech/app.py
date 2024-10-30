import time
from pathlib import Path

import ffmpeg
import yaml
from flask import Flask, g, jsonify, request, send_from_directory

from backend import ChatSpeech2Speech

app = Flask(__name__)

config = yaml.safe_load(Path(__file__).parent / "config.yaml")
service = ChatSpeech2Speech(config)


@app.route("/create_session")
async def init():
    return "init"


@app.route("/<session_id>/agent_can_speak")
async def agent_can_speak(session_id):
    return g.service.sessions[session_id].agent_can_speak


@app.route("/<session_id>/agent_can_speak")
async def agent_can_speak(session_id):
    return g.service.sessions[session_id].agent_can_speak


@app.route("/<session_id>/<filename>")
def hls_stream(session_id, filename):
    directory = g.service.sessions[session_id].workspace
    while not (directory / filename).exists():
        time.sleep(1)
    return send_from_directory(directory, filename)


@app.route("/<session_id>/transcript")
def transcript(session_id):
    return jsonify(g.service.sessions[session_id].transcript)


@app.route("/<session_id>/chat_messages")
def chat_messages(session_id):
    return jsonify(g.service.sessions[session_id].chat_messages)


@app.route("/<session_id>/user_audio", methods=["POST"])
def user_audio(session_id):
    audio_stream = ffmpeg.input("pipe:0")
    audio_stream = ffmpeg.output(audio_stream, g.service.sessions[session_id].user_audio_input_stream)
    ffmpeg.run(audio_stream, input=request.stream.read())
    return "OK"


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
