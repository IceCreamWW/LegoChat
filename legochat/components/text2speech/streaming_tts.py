import logging
import base64
import requests

import re

from legochat.components import Component, register_component

logger = logging.getLogger("legochat")


def extract_tts_text(text):
    punctuation = r"[，。！？,.!?]"
    for i in range(len(text), 5, -1):
        prefix = text[:i]
        if re.search(punctuation + r"$", prefix) and len(prefix) > 5:
            return prefix, text[i:]
    return "", text


@register_component("text2speech", "streaming_tts")
class StreamingTTSComponent(Component):

    def __init__(self, base_url):
        self.base_url = base_url
        self.sample_rate = 24000

    def setup(self):
        logger.info("StreamingTTSComponent setup")

    def tts(self, text):
        with requests.get(
            f"{self.base_url}/tts", params={"text": text}, stream=True
        ) as response:
            for chunk in response.iter_lines():
                if chunk:
                    chunk = chunk.decode("utf-8")
                    if chunk.startswith("data:"):
                        chunk = chunk[6:]  # remove the "data:" prefix
                        chunk = base64.b64decode(chunk)
                        logger.info(
                            f"Streaming TTS received chunk of size {len(chunk)} bytes"
                        )
                        yield chunk

    def process_func(self, text_fifo_path, audio_fifo_path, control_pipe=None):
        text = ""
        with open(text_fifo_path, "r", encoding="u8") as fifo_text, open(
            audio_fifo_path, "wb"
        ) as fifo_audio:
            while True:
                text_partial = fifo_text.read(5)
                if not text_partial:
                    break
                if control_pipe and control_pipe.poll():
                    signal = control_pipe.recv()
                    if signal == "interrupt":
                        logger.debug("Streaminmg TTS process interrupted")
                        break
                text += text_partial
                tts_text, text = extract_tts_text(text)
                if tts_text:
                    for chunk in self.tts(tts_text):
                        fifo_audio.write(chunk)
            if text:
                for chunk in self.tts(text):
                    fifo_audio.write(chunk)
        if control_pipe:
            control_pipe.close()
        logger.debug("Streaming TTS Process Finished")
        return 0


if __name__ == "__main__":
    pass
