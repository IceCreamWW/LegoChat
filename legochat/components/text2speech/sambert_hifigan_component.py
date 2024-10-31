import logging
import os

os.environ["MODELSCOPE_LOG_LEVEL"] = str(logging.ERROR)
import re
import time

from legochat.components import Component, register_component
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks


def extract_tts_text(text):
    punctuation = r"[，。！？,.!?]"
    for i in range(len(text), 5, -1):
        prefix = text[:i]
        if re.search(punctuation + r"$", prefix) and len(prefix) > 5:
            return prefix, text[i:]
    return "", text


@register_component("sambert_hifigan")
class SamBertHiFiGanComponent(Component):

    def __init__(self, model_name="damo/speech_sambert-hifigan_tts_zhiyan_emo_zh-cn_16k"):
        self.model_name = model_name
        self.wav_header_length = 44
        self.sample_rate = 16000

    def setup(self):
        return
        self.tts = pipeline(task=Tasks.text_to_speech, model=self.model_name)
        self.tts("启动")

    def process_func(self, text_fifo_path, audio_fifo_path, control_pipe=None):
        fd_text = os.open(text_fifo_path, os.O_RDWR | os.O_NONBLOCK)
        with open(fd_text, "r") as fifo_text:
            text_partial = fifo_text.read(5)
            if not text_partial:
                return
            if control_pipe and control_pipe.poll():
                signal = control_pipe.recv()
                if signal == "interrupt":
                    print("text2speech received interrupt signal")
                    return
            print("text2speech received text_partial: ", text_partial)
            time.sleep(1)
        os.close(fd_text)
        return 0

        text = ""
        with open(text_fifo_path, "r") as fifo_text, open(audio_fifo_path, "wb") as fifo_audio:
            while True:
                text_partial = fifo_text.read(5)
                if not text_partial:
                    break
                if control_pipe and control_pipe.poll():
                    signal = control_pipe.recv()
                    if signal == "interrupt":
                        break
                text += text_partial
                tts_text, text = extract_tts_text(text)
                if tts_text:
                    wav_bytes = self.tts(input=tts_text)[OutputKeys.OUTPUT_WAV]
                    fifo_audio.write(wav_bytes[self.wav_header_length :])
            if text:
                wav_bytes = self.tts(input=text)[OutputKeys.OUTPUT_WAV]
                fifo_audio.write(wav_bytes[self.wav_header_length :])
        return 0


if __name__ == "__main__":
    import os
    import threading
    from pathlib import Path

    fifo_path = Path("/tmp/my_fifo")
    # Create a FIFO
    if fifo_path.exists():
        fifo_path.unlink()
    os.mkfifo(fifo_path)

    def reader():
        with open(fifo_path, "r") as fifo:
            while True:
                data = fifo.readline()
                if not data:
                    break
                print(data)

    def writer():
        with open(fifo_path, "w") as fifo:
            for i in range(3):
                fifo.write(f"Hello from writer: {i}!\n")
                fifo.flush()
                time.sleep(2)

    # Start both reader and writer threads
    threading.Thread(target=reader).start()
    threading.Thread(target=writer).start()
