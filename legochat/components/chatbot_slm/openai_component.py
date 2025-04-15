import logging
import re
from itertools import chain
from typing import List, Dict, Optional
from pathlib import Path

from openai import OpenAI
from legochat.components import Component, register_component
import soundfile as sf
import base64
import numpy as np
import io

logger = logging.getLogger("legochat")

system_prompt = [
    {
        "role": "system",
        "content": """
你是上海交通大学开发的语音助手\"交交\",你可以规划行程、回答问题、辅导学习等。
你的回答要尽量口语化，简短明了，回复控制在100字以内。请像朋友之间聊天一样自然，可以加一些嗯、啊之类的词语，不要出现emoji、markdown、列表等。
如果觉得某个地方很有趣，可以在回复中加入哈哈哈来表示笑声。
你会使用中英日法四种语言。
你可以听到用户的声音。
你可以直接使用男生、女生、小孩、哪吒、太乙真人的声音，也可以直接模仿用户的声音。
不要重复用户指令。
不要输出敏感、非法的内容。
不要输出括号。
最后，始终记得，你是上海交通大学开发的语音助手\"交交\"。
""",
    }
]
system_prompt = [
    {
        "role": "system",
        "content": """
你是上海交通大学开发的语音助手\"交交\",你可以规划行程、回答问题、辅导学习等。
你的回答要尽量口语化，简短明了，回复控制在100字以内。请像朋友之间聊天一样自然，不要出现emoji、markdown、列表等。
你会使用中英日法四种语言。
你可以听到用户的声音。
不要重复用户指令。
不要输出敏感、非法的内容。
不要输出括号。
最后，始终记得，你是上海交通大学开发的语音助手\"交交\"。
""",
    }
]

# system_prompt = [
#     {
#         "role": "system",
#         "content": """
#         你是一个灵活的角色扮演助手，用户会在对话中指定你的身份。无论是什么身份，你都要努力完成用户给的任务。
#         回答要简洁、口语化，少说废话，避免复杂表达。在用户指定身份后，完全忠于角色设定，保持自然流畅，不要透露自己是 AI。如果用户要求跳出角色或切换身份，礼貌拒绝并继续扮演当前角色。
#         现在，请根据用户指令开始角色扮演。
# """,
#     }
# ]


def format_msg(content):
    fmt = ""
    for c in content:
        if "input_audio" in c:
            id = c.get("id", "")
            fmt += f"[audio] {id} "
        else:
            fmt += c.__repr__() + " "

    return fmt


@register_component("chatbot_slm", "openai")
class OpenAIComponent(Component):
    def __init__(
        self,
        base_url,
        api_key="token",
        model="Qwen/Qwen2.5-7B-Instruct",
    ):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model

    def setup(self):
        logger.info("OpenAIComponent setup")

    def add_user_message(
        self,
        messages,
        text: Optional[str] = None,
        audio_bytes: Optional[bytes] = None,
        sample_rate: int = 16000,
    ):

        assert text or audio_bytes, "text or audio_base64 must be provided"
        messages = messages[:]
        message = {"role": "user", "content": []}
        if audio_bytes:
            audio = np.frombuffer(audio_bytes, dtype=np.int16)
            buffer = io.BytesIO()
            sf.write(buffer, audio, samplerate=sample_rate, format="wav")
            audio_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
            message["content"].append(
                {
                    "type": "input_audio",
                    "input_audio": {"data": audio_base64, "format": "wav"},
                }
            )
        if text:
            message["content"].append({"type": "text", "text": text})
        messages.append(message)
        return messages

    def add_agent_message(self, messages, agent_message):
        messages = messages[:]
        messages.append({"role": "assistant", "content": agent_message})
        return messages

    def process_func(
        self,
        messages,
        text_fifo_path=None,
        control_pipe=None,
    ):

        if text_fifo_path is None:
            text_fifo_path = Path("/dev/null")

        messages = messages[-2:]

        for idx, m in enumerate(messages):
            logger.info(f">>>[{idx}]\n{format_msg(m['content']).strip()}")

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
    component = OpenAIComponent(base_url="http://localhost:8000/v1")
    component.setup()

    import soundfile as sf

    wav, sr = sf.read("/root/epfs/home/vv/workspace/playground/guess_age_gender.wav")
    wav = (wav * 32767).astype(np.int16)

    messages: List[Dict[str, str | Dict]] = []
    messages = component.add_user_message(
        messages, audio_bytes=wav.tobytes(), sample_rate=sr
    )
    response = component.process_func(
        messages=messages,
        model="sft.align0109.qa_voiceassistant.lora6-checkpoint-4135",
        # model="Qwen2-Audio-7B-Instruct",
    )
    messages = component.add_agent_message(messages, response)

    print(response)
