import logging
import time

from legochat.components import Component, register_component

logger = logging.getLogger("legochat")


@register_component("chatbot", "null")
class NullComponent(Component):
    def __init__(self, *args, **kwargs):
        pass

    def setup(self):
        return

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
        if text_fifo_path is None:
            text_fifo_path = Path("/dev/null")
        response = messages[-1]["content"]
        with open(text_fifo_path, "w") as fifo:
            fifo.write(response)
        return response
