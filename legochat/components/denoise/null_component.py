import logging
import re
from typing import List, Dict, Optional
from pathlib import Path

from openai import OpenAI
from legochat.components import Component, register_component
import soundfile as sf
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.fileio import File

logger = logging.getLogger("legochat")

@register_component("denoise", "null")
class NullComponent(Component):
    def __init__(self):
        pass

    def setup(self):
        pass

    def process_func(
        self,
        samples: bytes,
        prev_states: dict = None,
        end_of_stream: bool = False,
    ):
        return samples, prev_states


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.setLevel(logging.DEBUG)
    component = ZipEnhancerComponent()
    component.setup()

    import soundfile as sf

    wav, sr = sf.read("/root/epfs/home/vv/workspace/playground/guess_age_gender.wav")
    wav = (wav * 32767).astype(np.int16)
    wav_bytes = wav.tobytes()
    result, _ = component.process_func(wav_bytes)

    print(response)
