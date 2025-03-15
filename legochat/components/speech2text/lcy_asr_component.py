import sys
sys.path.append("/mnt/disk2/home/vv/workspace/lcy_translation/whisper35")
sys.path.append("/mnt/disk2/home/vv/workspace/lcy_translation")
import logging
import time
from datetime import datetime

import soundfile as sf
import torch
import numpy as np
from legochat.components import Component, register_component

from whisper35.pl_module import StreamingWhisper35Module
from whisper35.parse_yaml_args import parse_args_and_yaml
logger = logging.getLogger("legochat")


@register_component("speech2text", "lcy_asr")
class LcyAsrComponent(Component):
    def __init__(self):
        self.is_streaming = True # notify backend this asr is streaming model
        self.chunk_samples = 10240

    def setup(self):
        checkpoint_path='/mnt/disk2/home/vv/workspace/lcy_translation/joint.ckpt'
        cfg = parse_args_and_yaml(config_path="/mnt/disk2/home/vv/workspace/lcy_translation/whisper35.yaml")
        target_device = torch.device('cuda')

        module = StreamingWhisper35Module.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            cfg=cfg,
            map_location=target_device,
            strict=False,
        ).half().eval().to(target_device)

        self.model = module.model
        self.tokenizer = module.tokenizer
        logger.info("Lcy ASR model loaded")
        self.cache = {}

    def process_func(
        self,
        samples: bytes,
        prev_states: dict = None,
        end_of_stream: bool = False,
    ):
        if prev_states is not None:
            chunk_offset, text, shared_encode_cache, asr_encode_cache, decode_cache = self.cache[prev_states]
        else:
            chunk_offset = 0
            text = ""
            shared_encode_cache = None
            asr_encode_cache = None
            decode_cache = {'asr_out': None, 'prev_hyp': None, 'process_idx': None, 'cur_end_frame': None}
            prev_states = len(self.cache)
            self.cache[prev_states] = [chunk_offset, text, shared_encode_cache, asr_encode_cache, decode_cache]

        speech = np.frombuffer(samples, dtype=np.int16).astype(np.float32) / 32768.0
        speech_chunk = speech[chunk_offset:chunk_offset + self.chunk_samples]
        logger.debug(f"len of speech chunk: {len(speech_chunk)}")

        if end_of_stream:
            logger.info("end of stream asr")
            # sf.write(f"audio_chunk_{prev_states}.wav", speech, 16000)
            speech_chunk = speech[chunk_offset:]
            if len(speech_chunk) == 0:
                speech_chunk = np.zeros(8000)
        else:
            if len(speech_chunk) < self.chunk_samples:
                return text, prev_states

        # timestamp = datetime.now().strftime('%H%M%S.%f')
        # sf.write(f"audio_chunk_{prev_states}_{timestamp}.wav", speech_chunk, 16000)

        _, hyp, _, shared_encode_cache, asr_encode_cache, decode_cache = self.model.generate(
            speech_chunk,
            shared_encode_cache=shared_encode_cache,
            asr_encode_cache=asr_encode_cache,
            decode_cache=decode_cache,
            tokenizer=self.tokenizer,
            # lang_id='en',
            task='transcribe',
            use_lite=False,
            is_final=end_of_stream
        )
        if end_of_stream:
            del self.cache[prev_states]
        else:
            self.cache[prev_states] = [chunk_offset + self.chunk_samples, hyp, shared_encode_cache, asr_encode_cache, decode_cache]
        return hyp, prev_states


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
