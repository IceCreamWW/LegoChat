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

def create_wav_header(dataflow, sample_rate=16000, num_channels=1, bits_per_sample=16):
    """
    创建WAV文件头的字节串。

    :param dataflow: 音频bytes数据（以字节为单位）。
    :param sample_rate: 采样率，默认16000。
    :param num_channels: 声道数，默认1（单声道）。
    :param bits_per_sample: 每个样本的位数，默认16。
    :return: WAV文件头的字节串和音频bytes数据。
    """
    total_data_len = len(dataflow)
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    data_chunk_size = total_data_len
    fmt_chunk_size = 16
    riff_chunk_size = 4 + (8 + fmt_chunk_size) + (8 + data_chunk_size)

    # 使用 bytearray 构建字节串
    header = bytearray()

    # RIFF/WAVE header
    header.extend(b'RIFF')
    header.extend(riff_chunk_size.to_bytes(4, byteorder='little'))
    header.extend(b'WAVE')

    # fmt subchunk
    header.extend(b'fmt ')
    header.extend(fmt_chunk_size.to_bytes(4, byteorder='little'))
    header.extend((1).to_bytes(2, byteorder='little'))  # Audio format (1 is PCM)
    header.extend(num_channels.to_bytes(2, byteorder='little'))
    header.extend(sample_rate.to_bytes(4, byteorder='little'))
    header.extend(byte_rate.to_bytes(4, byteorder='little'))
    header.extend(block_align.to_bytes(2, byteorder='little'))
    header.extend(bits_per_sample.to_bytes(2, byteorder='little'))

    # data subchunk
    header.extend(b'data')
    header.extend(data_chunk_size.to_bytes(4, byteorder='little'))

    return bytes(header) + dataflow

@register_component("denoise", "zipenhancer")
class ZipEnhancerComponent(Component):
    def __init__(self):
        pass

    def setup(self):
        self.model = pipeline(
            Tasks.acoustic_noise_suppression,
            model='iic/speech_zipenhancer_ans_multiloss_16k_base'
        )
        dummy_samples = b'\x00' * 16000 * 2
        result = self.model(create_wav_header(dummy_samples, sample_rate=16000, num_channels=1, bits_per_sample=16))
        logger.info("ZipEnhancer setup")

    def process_func(
        self,
        samples: bytes,
        prev_states: dict = None,
        end_of_stream: bool = False,
    ):
        result = self.model(create_wav_header(samples, sample_rate=16000, num_channels=1, bits_per_sample=16))
        output = result['output_pcm']
        return output, None


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
