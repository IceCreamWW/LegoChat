import logging
import base64
import numpy as np
import soundfile as sf
from typing import Optional
import io
import time

from legochat.components import Component, register_component

logger = logging.getLogger("legochat")


@register_component("chatbot_slm", "null")
class NullComponent(Component):
    def __init__(self, *args, **kwargs):
        pass

    def setup(self):
        return

    def add_user_message(
        self,
        messages,
        text: Optional[str] = None,
        audio_bytes: Optional[bytes] = None,
        sample_rate: int = 16000,
    ):
        assert text or audio_bytes, "text or audio_base64 must be provided"
        message = {"role": "user", "content": []}
        if audio_bytes:
            audio = np.frombuffer(audio_bytes, dtype=np.int16)
            buffer = io.BytesIO()
            sf.write(buffer, audio, samplerate=sample_rate, format="wav")
            audio_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
            message["content"].append(
                {
                    "type": "input_audio",
                    "input_audio": {"data": audio_base64, "format": "wav"},
                }
            )
        if text:
            message["content"].append({"type": "text", "text": text})
        messages.append(message)
        return messages

    def add_agent_message(self, messages, agent_message):
        messages.append({"role": "assistant", "content": agent_message})
        return messages

    def process_func(
        self,
        messages,
        text_fifo_path=None,
        control_pipe=None,
        states=None,
    ):
        if text_fifo_path is None:
            text_fifo_path = Path("/dev/null")
        response = "我只会说这一句话"
        with open(text_fifo_path, "w") as fifo:
            fifo.write(response)
        return response
