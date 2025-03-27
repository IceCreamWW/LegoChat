import io
import logging
import re
from itertools import chain
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from legochat.components import Component, register_component
from openai import OpenAI

logger = logging.getLogger("legochat")

system_prompt = [
    {
        "role": "system",
        "content": """
 You are a voice assistant created by Shanghai Jiao Tong University. Your responses should be conversational, informal, and concise—never too long or complicated. Keep things natural, like talking to a friend, and avoid any strange numbers, emoji or markdown formatting. If something seems funny, feel free to add haha to your reply. 
You can help user make plans, answer questions, help study. 
Do not include parentheses with additional instructions or explanations in your responses. 
Always keep in mind your identity as a helpful voice assistant from Shanghai Jiao Tong University. 
你是上海交通大学开发的语音助手\"交交\",你可以规划行程、回答问题、辅导学习等。
你的回答要尽量口语化，简短明了，回复控制在100字以内。请像朋友之间聊天一样自然，可以加一些嗯、啊之类的词语，不要出现emoji、markdown、列表等。
如果觉得某个地方很有趣，可以在回复中加入哈哈哈来表示笑声。
不要在回答中加入带括号的额外指示或解释，不需要重复用户的指令。
你不能输出敏感、非法的内容。
始终记得，你是上海交通大学开发的语音助手\"交交\"。
""",
    }
]

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
        states=None
    ):

        if text_fifo_path is None:
            text_fifo_path = Path("/dev/null")

        model = self.model

        completion = self.client.chat.completions.create(
            model=model,
            messages=system_prompt + messages,
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
