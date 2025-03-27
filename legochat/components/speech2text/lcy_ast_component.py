import sys
import requests
import logging
import time
from datetime import datetime

from legochat.components import Component, register_component
from transformers import AutoTokenizer

logger = logging.getLogger("legochat")


@register_component("speech2text", "lcy_ast")
class LcyAstComponent(Component):
    def __init__(self, base_url):
        self.base_url = base_url
        self.is_streaming = True # notify backend this asr is streaming model
        self.chunk_samples = 10240
        self.cache = {}

    def setup(self):
        logger.info("Lcy AST model loaded")

    def process_func(
        self,
        samples: bytes,
        end_of_stream: bool = False,
        states = None,
    ):

        states_ = states
        session_id = states["session_id"]

        if session_id not in self.cache:
            self.cache[session_id] = {}

        states = self.cache[session_id]
        chunk_offset = states.get("chunk_offset", 0)
        text = states.get("text", "")

        chunk_bytes = samples[chunk_offset:]

        if end_of_stream:
            logger.info("end of stream ast")
            chunk_bytes += b"\x00" * (self.chunk_samples * 2 - len(chunk_bytes))
            del self.cache[session_id]
        else:
            if len(chunk_bytes) < self.chunk_samples * 2:
                return text, states_

        response = requests.post(f"{self.base_url}/ast/{session_id}/{int(end_of_stream)}", data=chunk_bytes, headers={"Content-Type": "application/octet-stream"})
        states["chunk_offset"] = len(samples)
        text = response.json()["result"]
        states["text"] = text
        return text, states_


if __name__ == "__main__":
    import numpy as np
    import soundfile as sf

    audio_file = "examples/audio.wav"
    data, sr = sf.read(audio_file)
    data = (data * 32768.0).astype(np.int16).tobytes()
    import time

    component = ParaformerComponent(punctuation=True)
    component.setup()
    duration = len(data) / sr / 2
    for i in range(10):
        start = time.time()
        results, states = component.process_func(samples=data)
        end = time.time()
        print(f"rtf: {(end - start) / duration}")
    print(results)
