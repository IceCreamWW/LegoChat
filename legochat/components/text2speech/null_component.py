import base64
import json
import logging
import re

from uuid import uuid4
import requests
from legochat.components import Component, register_component
from openai import OpenAI

logger = logging.getLogger("legochat")


@register_component("text2speech", "null")
class NullComponent(Component):
    def __init__(self):
        self.sample_rate = 16000

    def setup(self):
        logger.info("StreamingTTSComponent setup")

    def process_func(self, text_fifo_path, audio_fifo_path, control_pipe=None, control_params=None):
        text = ""
        with open(text_fifo_path, "r", encoding="u8") as fifo_text, open(
            audio_fifo_path, "wb"
        ) as fifo_audio:
            while True:
                text_partial = fifo_text.read(5)
                if not text_partial:
                    break
            fifo_audio.write(b"\x00" * 32000)
        return 0


if __name__ == "__main__":
    pass
