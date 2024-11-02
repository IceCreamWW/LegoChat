import logging
import os
import threading
import time

from legochat.components import Component, register_component
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          TextIteratorStreamer)

logger = logging.getLogger("legochat")


@register_component("dummy_chatbot")
class DummyChatbotComponent(Component):
    def __init__(self, *args, **kwargs):
        pass

    def setup(self):
        return

    @property
    def system_prompt(self):
        messages = [
            {
                "role": "system",
                "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant. Limit your response to as short as possible.",
            },
        ]
        return messages[:]

    @property
    def pending_token(self):
        return ""

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
        text_fifo_path,
        control_pipe=None,
    ):
        response = "测试用这个回复," * 5
        with open(text_fifo_path, "w") as fifo:
            for c in response:
                logger.info("sending", c)
                fifo.write(c)
                fifo.flush()
                time.sleep(0.5)
                if control_pipe and control_pipe.poll():
                    signal = control_pipe.recv()
                    if signal == "interrupt":
                        logger.info("chatbot received interrupt signal")
                        break
        logger.info("chatbot sent all response")
        return response
