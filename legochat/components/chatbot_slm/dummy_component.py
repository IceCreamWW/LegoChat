import logging
import soundfile as sf
import numpy as np
import io
import base64

from typing import Optional
import time

from legochat.components import Component, register_component

logger = logging.getLogger("legochat")


@register_component("chatbot_slm", "dummy")
class DummyComponent(Component):
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
        **kwargs,
    ):
        response = "测试用这个回复," * 5
        with open(text_fifo_path, "w") as fifo:
            for c in response:
                logger.info(
                    f"sending {c}",
                )
                fifo.write(c)
                fifo.flush()
                time.sleep(0.2)
                if control_pipe and control_pipe.poll():
                    signal = control_pipe.recv()
                    if signal == "interrupt":
                        logger.info("chatbot received interrupt signal")
                        break
        logger.info("chatbot sent all response")
        return response
