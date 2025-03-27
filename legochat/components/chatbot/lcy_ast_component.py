import io
import requests
import logging
import re
from itertools import chain
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from legochat.components import Component, register_component
from openai import OpenAI

logger = logging.getLogger("legochat")


@register_component("chatbot", "lcy_ast")
class OpenAIComponent(Component):
    def __init__(
        self,
        base_url,
    ):
        self.base_url = base_url

    def setup(self):
        logger.info("Lcy AST setup")

    def add_user_message(self, messages, user_message):
        new_messages = messages[:]
        new_messages.append({"role": "user", "content": user_message})
        return new_messages

    def add_agent_message(self, messages, agent_message):
        new_messages = messages[:]
        new_messages.append({"role": "assistant", "content": agent_message})
        return new_messages

    def process_func(
        self,
        messages,
        text_fifo_path=None,
        control_pipe=None,
        states=None,
    ):
        session_id = states["session_id"]
        response = requests.get(f"{self.base_url}/ast/{session_id}")
        response = response.json()["result"]

        if text_fifo_path is None:
            text_fifo_path = Path("/dev/null")

        with open(text_fifo_path, "w") as fifo:
            fifo.write(response)
            fifo.flush()

        if control_pipe:
            control_pipe.close()
        return response, states


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.setLevel(logging.DEBUG)
    component = OpenAIComponent(base_url="http://localhost:7000/v1")
    component.setup()

    import soundfile as sf

    messages: List[Dict[str, str | Dict]] = []
    messages = component.add_user_message(messages, "hello")
    response = component.process_func(
        messages=messages,
        # model="Qwen2-Audio-7B-Instruct",
    )
    messages = component.add_agent_message(messages, response)

    print(response)
