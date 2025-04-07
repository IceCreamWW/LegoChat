import base64
import json
import logging
import re

from uuid import uuid4
import requests
from legochat.components import Component, register_component
from openai import OpenAI

logger = logging.getLogger("legochat")


@register_component("diarization", "null")
class NullComponent(Component):
    def __init__(self):
        self.sample_rate = 16000

    def setup(self):
        logger.info("diarization setup")

    def is_diarizaition(
        self, audio_bytes: bytes, transcript = None
    ):
        return False

    def process_func(self, session_id, audio_id, audio_bytes, transcript="")  -> dict[str, int]:
        return {}


if __name__ == "__main__":
    pass
