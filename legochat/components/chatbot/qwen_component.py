import logging
import os
import threading
import time
from pathlib import Path

from legochat.components import Component, register_component
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          TextIteratorStreamer)

logger = logging.getLogger("legochat")


class StoppableTextIteratorStreamer(TextIteratorStreamer):
    class Interrupt(Exception):
        def __init__(self, message="Generation was interrupted."):
            super().__init__(message)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._stop_event = threading.Event()

    def put(self, value):
        if self._stop_event.is_set():
            self.text_queue.put(self.stop_signal, timeout=self.timeout)
            raise self.Interrupt
        else:
            super().put(value)


@register_component("qwen")
class QwenComponent(Component):
    def __init__(self, model_name="Qwen/Qwen2.5-7B-Instruct"):
        self.model_name = model_name

    def setup(self):
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype="auto", device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        logger.info("Qwen model loaded")

    @property
    def system_prompt(self):

        prompt = """
You are an intelligent chatbot designed to assist the user based on transcribed speech text. 

Please follow these guidelines:

1. Handle Transcription Errors: If the user’s message has minor transcription errors, try to interpret the meaning based on context. Make a reasonable attempt to respond as accurately as possible.

2. Meaningless or Noisy Inputs: If the transcription contains random noises or appears nonsensical (e.g., jumbled words or letters with no coherent meaning), respond with [pd], indicating you're waiting for a clearer user input.

3. Partial Messages: When the user input is incomplete, like "今天是" or "这样的话," reply with [pd] to prompt them for further clarification.

4. Acknowledge Responses: If the user acknowledges your response with phrases like "了解了" or "这样啊," reply with [pd] to indicate you’re awaiting additional input.

5. Repetitive Responses: When the user sends identical messages in succession, such as "谢谢," respond normally to the first instance but use [pd] for any immediate, repeated responses.

6. important! never append or insert [pd] to a respond. [pd] should always be a single response.

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

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        streamer = StoppableTextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        generation_kwargs = dict(model_inputs, streamer=streamer, max_new_tokens=512)
        thread = threading.Thread(target=self.generate, kwargs=generation_kwargs)
        thread.start()

        if text_fifo_path is None:
            text_fifo_path = Path("/dev/null")

        response = ""
        with open(text_fifo_path, "w") as fifo:
            for response_partial in streamer:
                if control_pipe and control_pipe.poll():
                    try:
                        signal = control_pipe.recv()
                    except Exception as e:
                        logger.error(e)
                        signal = "interrupt"
                    if signal == "interrupt":
                        streamer._stop_event.set()
                        thread.join()
                        logger.debug("Qwen process interrupted")
                        break
                fifo.write(response_partial)
                fifo.flush()
                response += response_partial
                if self.pending_token and self.pending_token in response_partial:
                    break

        if control_pipe:
            control_pipe.close()
        return response

    @property
    def pending_token(self):
        return "[pd]"

    def generate(self, **kwargs):
        try:
            self.model.generate(**kwargs)
        except StoppableTextIteratorStreamer.Interrupt as e:
            logger.info(e)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.setLevel(logging.DEBUG)
    component = QwenComponent()
    component.setup()
    messages = component.system_prompt

    messages = component.add_user_message(messages, "简单介绍一下音频-文本大模型")
    response = component.process_func(messages=messages)
    messages = component.add_agent_message(messages, response)

    messages = component.add_user_message(messages, "谢谢")
    response = component.process_func(messages=messages)
    messages = component.add_agent_message(messages, response)

    messages = component.add_user_message(messages, "嗯嗯")
    response = component.process_func(messages=messages)
    messages = component.add_agent_message(messages, response)

    messages = component.add_user_message(messages, "这样的话")
    response = component.process_func(messages=messages)
    messages = component.add_agent_message(messages, response)

    messages = component.add_user_message(messages, "再简单介绍一下神经网络吧")
    response = component.process_func(messages=messages)
    messages = component.add_agent_message(messages, response)

    print(messages)
