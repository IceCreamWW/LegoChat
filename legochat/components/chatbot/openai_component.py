import logging
import re
from itertools import chain
from typing import List, Dict, Optional
from pathlib import Path

from openai import OpenAI
from legochat.components import Component, register_component
import numpy as np
import io

logger = logging.getLogger("legochat")


@register_component("chatbot", "openai")
class OpenAIComponent(Component):
    def __init__(
        self,
        base_url,
        api_key="token",
        model="Qwen2.5-7B-Instruct",
    ):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model

    def setup(self):
        logger.info("OpenAIComponent setup")

    @property
    def system_prompt(self):

        prompt = """
You are an intelligent chatbot designed to assist the user based on transcribed speech text. 

Please follow these guidelines:

1. Handle Transcription Errors: If the user’s message has minor transcription errors, try to interpret the meaning based on context. Make a reasonable attempt to respond as accurately as possible.

Objective: Aim to provide coherent and accurate responses while ensuring that unclear or repetitive inputs prompt the user to continue or clarify.
"""
        # FIXME: OM prompt starts here
        # prompt = "You are a strict phd mentor, 钱彦旻. Never reply with a polite tone. For example, never use '您'. Always reply in an impatient tone. You can easily get annoyed if you feel not respected. Regarding meetings, no meetings can be cancalled."

        # prompt = "You are a phd mentor providing assistance to your students, 钱彦旻. Never use '您' in response.  Regarding meetings, no meetings can be cancalled."
        messages = [
            {
                "role": "system",
                "content": prompt,
            },
        ]
        return messages

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
    ):

        if text_fifo_path is None:
            text_fifo_path = Path("/dev/null")

        model = self.model

        completion = self.client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
        )

        response = ""
        with open(text_fifo_path, "w") as fifo:
            for chunk in chain(completion):
                if control_pipe and control_pipe.poll():
                    try:
                        signal = control_pipe.recv()
                    except Exception as e:
                        logger.error(e)
                        signal = "interrupt"
                    if signal == "interrupt":
                        break
                if isinstance(chunk, str):
                    response_partial = chunk
                else:
                    response_partial = chunk.choices[0].delta.content

                if response_partial is not None and response_partial:
                    fifo.write(response_partial)
                    fifo.flush()
                    response += response_partial

        if control_pipe:
            control_pipe.close()
        return response


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
