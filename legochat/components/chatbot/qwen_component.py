import logging
import os
import threading
import time

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
        return
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype="auto", device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    @property
    def system_prompt(self):
        messages = [
            {
                "role": "system",
                "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant. Limit your response to as short as possible.",
            },
        ]
        return messages[:]

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
        print("in chatbot")
        response = "测试用这个回复," * 5
        with open(text_fifo_path, "w") as fifo:
            for c in response:
                print("sending", c)
                fifo.write(c)
                fifo.flush()
                time.sleep(0.01)
                if control_pipe and control_pipe.poll():
                    signal = control_pipe.recv()
                    if signal == "interrupt":
                        logger.info("chatbot received interrupt signal")
                        break
        logger.info("chatbot sent all response")
        return response

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

        response = ""
        with open(text_fifo_path, "w") as fifo:
            for response_partial in streamer:
                if control_pipe and control_pipe.poll():
                    signal = control_pipe.recv()
                    if signal == "interrupt":
                        streamer._stop_event.set()
                        thread.join()
                        break
                fifo.write(response_partial)
                fifo.flush()
                response += response_partial

        return response

    def generate(self, **kwargs):
        try:
            self.model.generate(**kwargs)
        except StoppableTextIteratorStreamer.Interrupt as e:
            logger.info(e)


if __name__ == "__main__":
    logger.basicConfig(level=logging.INFO)
    prompt = "简单介绍一下音频-文本大模型"
    messages = [
        {
            "role": "system",
            "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant. Limit your response to as short as possible.",
        },
        {"role": "user", "content": prompt},
    ]
    component = QwenComponent()
    result = component.process(messages)
    for new_text in result:
        print(new_text)
