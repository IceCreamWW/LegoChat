from threading import Thread

from legochat.components import Component, register_component
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          TextIteratorStreamer)


@register_component("qwen")
class QwenComponent(Component):
    def __init__(self, model_name="Qwen/Qwen2.5-7B-Instruct"):
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def process(self, messages, end_of_stream=False):
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(model.device)

        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        generation_kwargs = dict(model_inputs, streamer=streamer, max_new_tokens=512)
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        for new_text in streamer:
            yield new_text


prompt = "简单介绍一下音频-文本大模型"
messages = [
    {
        "role": "system",
        "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant. Limit your response to as short as possible.",
    },
    {"role": "user", "content": prompt},
]
