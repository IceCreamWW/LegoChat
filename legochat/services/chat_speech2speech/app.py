import asyncio
import logging

logging.getLogger("legochat").setLevel(logging.INFO)
logging.getLogger("werkzeug").setLevel(logging.WARNING)
import threading
import time
from pathlib import Path
from uuid import uuid4

import yaml
from flask import Flask, jsonify, render_template, request, send_from_directory
from legochat.utils.event import EventEnum
from legochat.utils.stream import FIFOAudioIOStream

from backend import ChatSpeech2Speech

background_loop = asyncio.new_event_loop()


def start_background_loop(loop):
    asyncio.set_event_loop(loop)
    loop.run_forever()


threading.Thread(target=start_background_loop, args=(background_loop,), daemon=True).start()


app = Flask(__name__)

# Load configuration
config = yaml.safe_load((Path(__file__).parent / "config.yaml").read_text())

# Initialize the service
service = ChatSpeech2Speech(config)
service.run()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/start_session")
def start_session():
    allow_vad_interrupt = request.args.get("allow_vad_interrupt", "true").lower() == "true"
    allow_vad_eot = request.args.get("allow_vad_eot", "true").lower() == "true"
    sample_rate = int(request.args.get("sample_rate", 16000))  # default to 16kHz if not specified

    session_id = uuid4().hex
    workspace = Path(f"workspace/{session_id}")
    workspace.mkdir(parents=True, exist_ok=True)

    user_audio_stream = FIFOAudioIOStream(mode="rw", sample_rate_w=sample_rate, sample_rate_r=16000)
    agent_audio_output_stream = FIFOAudioIOStream(mode="c", sample_rate_w=service.text2speech.sample_rate)

    # Start session asynchronously
    asyncio.run_coroutine_threadsafe(
        service.start_session(
            session_id=session_id,
            workspace=workspace,
            user_audio_input_stream=user_audio_stream,
            agent_audio_output_stream=agent_audio_output_stream,
            allow_vad_interrupt=allow_vad_interrupt,
            allow_vad_eot=allow_vad_eot,
        ),
        background_loop,
    )
    agent_audio_output_stream.stream_to_m3u8(workspace / "agent.m3u8")
    return jsonify({"session_id": session_id})


@app.route("/<session_id>/test")
async def test_write_audio(session_id):
    await service.sessions[session_id].agent_audio_output_stream.write(8192 * b"\x00")
    return "OK"


@app.route("/<session_id>/agent_can_speak")
def agent_can_speak(session_id):
    can_speak = service.sessions.get(session_id).agent_can_speak
    return jsonify({"agent_can_speak": can_speak})


@app.route("/<session_id>/assets/<filename>")
def get_session_file(session_id, filename):
    directory = service.sessions[session_id].workspace.absolute()
    while not (directory / filename).exists():
        time.sleep(1)
    print(f"Sending {directory / filename}")
    return send_from_directory(directory, filename)


@app.route("/<session_id>/transcript")
def transcript(session_id):
    transcript = service.sessions[session_id].transcript
    return jsonify(transcript)


@app.route("/<session_id>/chat_messages")
def chat_messages(session_id):
    chat_messages = service.sessions[session_id].chat_messages
    return jsonify(chat_messages)


@app.route("/<session_id>/user_audio", methods=["POST"])
async def user_audio(session_id):
    data = request.data
    await service.sessions[session_id].user_audio_input_stream.write(data)
    return "OK", 200


@app.route("/<session_id>/interrupt", methods=["POST"])
async def interrupt(session_id):
    asyncio.run_coroutine_threadsafe(
        service.sessions[session_id].event_bus.emit(EventEnum.INTERRUPT, sender="user"), background_loop
    )
    return "Interrupted", 200


@app.route("/<session_id>/end_of_turn", methods=["POST"])
async def end_of_turn(session_id):
    # await it in backgroud loop
    asyncio.run_coroutine_threadsafe(
        service.sessions[session_id].event_bus.emit(EventEnum.END_OF_TURN, sender="user"), background_loop
    )
    return "End of Turn Noted", 200


@app.route("/<session_id>/test")
async def test(session_id):
    await service.sessions[session_id].agent_audio_output_stream.write(8192 * b"\x00")
    return "OK"


if __name__ == "__main__":
    certs = ("certs/cert.pem", "certs/key.pem")
    app.run(host="0.0.0.0", port=5555, use_reloader=False, ssl_context=certs)
