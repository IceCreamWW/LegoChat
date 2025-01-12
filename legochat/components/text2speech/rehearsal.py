import logging
import soundfile as sf
from typing import List, Dict
import os

os.environ["MODELSCOPE_LOG_LEVEL"] = str(logging.ERROR)
import re
import time

from legochat.components import Component, register_component
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

logger = logging.getLogger("legochat")


def extract_tts_text(text):
    punctuation = r"[，。！？,.!?]"
    for i in range(len(text), 5, -1):
        prefix = text[:i]
        if re.search(punctuation + r"$", prefix) and len(prefix) > 5:
            return prefix, text[i:]
    return "", text


@register_component("text2speech", "rehearsal")
class RehearsalComponent(Component):

    def __init__(self, responses: List[Dict], overhead_seconds: float = 0.5):
        self.wav_header_length = 44
        self.sample_rate = 16000
        self.responses = responses

    def setup(self):
        logger.info("tts setup")

    def process_func(
        self,
        text_fifo_path,
        audio_fifo_path,
        control_pipe=None,
        scene=None,
        ith_message=0,
    ):
        audio_file = self.responses[scene][ith_message]["audio"]
        audio, sample_rate = sf.read(audio_file)
        assert sample_rate == 16000
        wav_bytes = (audio * (2**15)).astype("int16").tobytes()
        with open(text_fifo_path, "r", encoding="u8") as fifo_text, open(
            audio_fifo_path, "wb"
        ) as fifo_audio:
            fifo_audio.write(wav_bytes)
            fifo_text.read()

        if control_pipe:
            control_pipe.close()
        logger.debug("TTS process finished")
        return 0


if __name__ == "__main__":
    pass
