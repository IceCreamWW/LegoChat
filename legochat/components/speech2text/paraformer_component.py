# pip3 install -U funasr-onnx
import numpy as np
from funasr_onnx import CT_Transformer, Paraformer
from legochat.components import Component, register_component


@register_component("paraformer")
class ParaformerComponent(Component):
    def __init__(self, punctuation=False):
        self.punctuation = punctuation

    def setup(self):
        self.speech2text_model = Paraformer(
            "damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch", batch_size=1, quantize=True
        )
        self.punctuation_model = (
            CT_Transformer("damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch") if self.punctuation else None
        )

    def process_func(
        self,
        samples: bytes,
        prev_states: dict = None,
        end_of_stream: bool = False,
    ):
        assert prev_states is None, "Paraformer stateful processing is not supported yet"
        samples = np.frombuffer(samples, dtype=np.int16)
        result = self.speech2text_model(samples)
        text = result[0]["preds"][0]
        if self.punctuation_model:
            text = self.punctuation_model(text)[0]
        return text, None


if __name__ == "__main__":
    import numpy as np
    import soundfile as sf

    audio_file = "examples/audio.wav"
    data, sr = sf.read(audio_file)
    data = (data * 32768.0).astype(np.int16).tobytes()
    component = ParaformerComponent(punctuation=True)
    results, states = component.process(data, None, True)
    breakpoint()
    print(results)
