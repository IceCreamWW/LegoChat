import logging
from typing import List, Dict
import soundfile as sf
import numpy as np
import io
import base64

from typing import Optional
import time

from legochat.components import Component, register_component

logger = logging.getLogger("legochat")


@register_component("chatbot_slm", "rehearsal")
class RehearsalComponent(Component):
    def __init__(
        self,
        responses: List[Dict],
        overhead_seconds: float = 0.2,
        char_delay_seconds: float = 0.01,
    ):
        self.responses = responses
        self.overhead_seconds = overhead_seconds
        self.char_delay_seconds = char_delay_seconds

    def setup(self):
        logger.info("chatbot setup")
        return

    def add_user_message(
        self,
        messages,
        text: Optional[str] = None,
        audio_bytes: Optional[bytes] = None,
        sample_rate: int = 16000,
    ):

        assert text or audio_bytes, "text or audio_base64 must be provided"
        new_messages = messages[:]
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
        new_messages.append(message)
        return new_messages

    def add_agent_message(self, messages, agent_message):
        new_messages = messages[:]
        new_messages.append({"role": "assistant", "content": agent_message})
        return new_messages

    def process_func(
        self,
        messages,
        text_fifo_path,
        control_pipe=None,
        scene=None,
        ith_message=0,
        **kwargs,
    ):
        time.sleep(self.overhead_seconds)
        response = self.responses[scene][ith_message]["text"]
        with open(text_fifo_path, "w") as fifo:
            for c in response:
                fifo.write(c)
                fifo.flush()
                time.sleep(self.char_delay_seconds)
                if control_pipe and control_pipe.poll():
                    signal = control_pipe.recv()
                    if signal == "interrupt":
                        logger.info("chatbot received interrupt signal")
                        break
        logger.info("chatbot sent all response")
        return response
