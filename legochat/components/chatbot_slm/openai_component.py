import logging
from typing import List, Dict, Optional
from pathlib import Path

from openai import OpenAI
from legochat.components import Component, register_component
import soundfile as sf
import base64
import numpy as np
import io

logger = logging.getLogger("legochat")


@register_component("chatbot_slm", "openai")
class OpenAIComponent(Component):
    def __init__(self, base_url, api_key="token", model="Qwen/Qwen2.5-32B-Instruct"):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model

    def setup(self):
        logger.info("OpenAIComponent setup")
        pass

    def add_user_message(
        self,
        messages,
        text: Optional[str] = None,
        audio_bytes: Optional[bytes] = None,
        sample_rate: int = 16000,
    ):

        assert text or audio_bytes, "text or audio_base64 must be provided"
        new_messages = messages[:]
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
        new_messages.append(message)
        return new_messages

    def add_agent_message(self, messages, agent_message):
        new_messages = messages[:]
        new_messages.append({"role": "assistant", "content": agent_message})
        return new_messages

    # print(completion.choices[0].message)
    def process_func(
        self,
        messages,
        model=None,
        text_fifo_path=None,
        control_pipe=None,
    ):

        if text_fifo_path is None:
            text_fifo_path = Path("/dev/null")

        model = model if model else self.model

        completion = self.client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
        )

        response = ""
        with open(text_fifo_path, "w") as fifo:
            for chunk in completion:
                if control_pipe and control_pipe.poll():
                    try:
                        signal = control_pipe.recv()
                    except Exception as e:
                        logger.error(e)
                        signal = "interrupt"
                    if signal == "interrupt":
                        break
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
    component = OpenAIComponent(base_url="http://localhost:11000/v1")
    component.setup()

    import soundfile as sf

    wav, sr = sf.read("/root/epfs/home/vv/workspace/playground/guess_age_gender.wav")
    wav = (wav * 32767).astype(np.int16)

    messages: List[Dict[str, str | Dict]] = []
    messages = component.add_user_message(
        messages, audio_bytes=wav.tobytes(), sample_rate=sr
    )
    response = component.process_func(
        messages=messages, model="Qwen/Qwen2-Audio-7B-Instruct"
    )
    messages = component.add_agent_message(messages, response)

    print(response)
