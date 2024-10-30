import logging
import time
from pathlib import Path
from uuid import uuid4

import ffmpeg
import yaml
from flask import (Flask, g, jsonify, render_template, request,
                   send_from_directory)
from legochat.utils.event import EventEnum
from legochat.utils.stream import FIFOAudioIOStream, M3U8AudioOutputStream

from backend import ChatSpeech2Speech

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

config = yaml.safe_load((Path(__file__).parent / "config.yaml").read_text())

service = ChatSpeech2Speech(config)
service.run()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/create_session")
async def init():
    breakpoint()
    allow_vad_interrupt = request.args.get("allow_vad_interrupt", True)
    allow_vad_eot = request.args.get("allow_vad_eot", True)
    sample_rate = int(request.args.get("sample_rate", 16000))  # default to 16kHz if not specified

    session_id = uuid4().hex
    workspace = Path(f"workspace/{session_id}")
    user_audio_input_stream = FIFOAudioIOStream()
    agent_audio_output_stream = M3U8AudioOutputStream(
        workspace / "agent" / "playlist.m3u8",
    )
    session = service.create_session(
        session_id=session_id,
        workspace=workspace,
        user_audio_input_stream=user_audio_input_stream,
        agent_audio_output_stream=agent_audio_output_stream,
        allow_vad_interrupt=allow_vad_interrupt,
        allow_vad_eot=allow_vad_eot,
    )
    return session.session_id


@app.route("/<session_id>/agent_can_speak")
async def agent_can_speak(session_id):
    return service.sessions[session_id].agent_can_speak


@app.route("/<session_id>/<filename>")
def hls_stream(session_id, filename):
    directory = service.sessions[session_id].workspace
    while not (directory / filename).exists():
        time.sleep(1)
    return send_from_directory(directory, filename)


@app.route("/<session_id>/transcript")
def transcript(session_id):
    return jsonify(service.sessions[session_id].transcript)


@app.route("/<session_id>/chat_messages")
def chat_messages(session_id):
    return jsonify(service.sessions[session_id].chat_messages)


@app.route("/<session_id>/user_audio", methods=["POST"])
async def user_audio(session_id):
    data = request.data
    await service.sessions[session_id].user_audio_input_stream.write(data)
    return "OK"


@app.route("/<session_id>/interrupt", methods=["POST"])
async def interrupt(session_id):
    await service.sessions[session_id].event_bus.trigger_event(EventEnum.INTERRUPT, sender="user")
    return "Interrupted", 200


@app.route("/<session_id>/end_of_turn", methods=["POST"])
async def end_of_turn(session_id):
    await service.sessions[session_id].event_bus.trigger_event(EventEnum.END_OF_TURN, sender="user")
    return "End of Turn Noted", 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5555, use_reloader=False)
