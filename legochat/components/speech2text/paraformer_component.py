import logging
import time

import numpy as np
from funasr_onnx import CT_Transformer, Paraformer
from legochat.components import Component, register_component

logger = logging.getLogger("legochat")


@register_component("speech2text", "paraformer")
class ParaformerComponent(Component):
    def __init__(self, punctuation=True):
        self.punctuation = punctuation

    def setup(self):
        self.speech2text_model = Paraformer(
            "damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
            batch_size=1,
            quantize=True,
        )
        self.punctuation_model = (
            CT_Transformer("damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch")
            if self.punctuation
            else None
        )
        logger.info("Paraformer model loaded")

    def process_func(
        self,
        samples: bytes,
        prev_states: dict = None,
        end_of_stream: bool = False,
    ):
        start = time.time()
        assert (
            prev_states is None
        ), "Paraformer stateful processing is not supported yet"
        samples = np.frombuffer(samples, dtype=np.int16)
        result = self.speech2text_model(samples)
        if not result or not result[0]["preds"]:
            return "", None
        text = result[0]["preds"][0]
        if text.strip() == "":
            return "", None
        if self.punctuation_model:
            text = self.punctuation_model(text)[0]
        end = time.time()

        input_seconds = len(samples) / 16000
        inference_seconds = end - start

        logger.debug(
            f"Paraformer cost {inference_seconds:.3f}s, rtf: {inference_seconds / input_seconds:.3f}"
        )
        return text, None


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
