import logging
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


@register_component("text2speech", "sambert_hifigan")
class SamBertHiFiGanComponent(Component):

    def __init__(
        self, model_name="damo/speech_sambert-hifigan_tts_zhiyan_emo_zh-cn_16k"
    ):
        self.model_name = model_name
        self.wav_header_length = 44
        self.sample_rate = 16000

    def setup(self):
        self.tts = pipeline(task=Tasks.text_to_speech, model=self.model_name)
        self.tts("启动")
        logger.info("SamBert-HiFi-GAN model loaded")

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
                        logger.debug("SamBert-HiFi-GAN process interrupted")
                        break
                text += text_partial
                tts_text, text = extract_tts_text(text)
                if tts_text:
                    logger.info(f"SamBert-HiFi-GAN synthesize [{tts_text}]")
                    start = time.time()
                    wav_bytes = self.tts(input=tts_text)[OutputKeys.OUTPUT_WAV]
                    end = time.time()
                    output_seconds = (
                        len(wav_bytes[self.wav_header_length :]) / 2 / self.sample_rate
                    )
                    infernece_seconds = end - start
                    logger.debug(
                        f"SamBert-HiFi-GAN synthesize [{tts_text}][{output_seconds:.3f}s]; cost {infernece_seconds:.3f}s, rtf: {infernece_seconds / output_seconds:.3f}"
                    )
                    fifo_audio.write(wav_bytes[self.wav_header_length :])
            if text:
                wav_bytes = self.tts(input=text)[OutputKeys.OUTPUT_WAV]
                fifo_audio.write(wav_bytes[self.wav_header_length :])
        if control_pipe:
            control_pipe.close()
        logger.debug("SamBert-HiFi-GAN process finished")
        return 0


if __name__ == "__main__":
    pass
