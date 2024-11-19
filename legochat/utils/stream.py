import asyncio
import errno
import fcntl
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.signal import resample

logger = logging.getLogger("legochat")


class AudioInputStream:
    async def read(self, size: int = -1):
        raise NotImplementedError


class AudioOutputStream:
    async def write(self, data):
        raise NotImplementedError


class FIFOAudioIOStream(AudioInputStream, AudioOutputStream):
    """A class for reading and writing audio data to a FIFO file.

    Attributes:
        fifo_path (str): Path to the FIFO file.
        sample_rate_r (int): Sample rate of the output audio stream.
        sample_rate_w (int): Sample rate of the input audio stream.
    Note:
        If sample_rate_r and sample_rate_w are specified, the input audio stream will be resampled from sample_rate_w to sample_rate_r.
    """

    def __init__(
        self,
        fifo_path: Optional[str | Path] = None,
        sample_rate_r: int = None,
        sample_rate_w: int = None,
        m3u8_path: Optional[str | Path] = None,
    ):
        self.sample_rate_r = sample_rate_r
        self.sample_rate_w = sample_rate_w
        self.m3u8_path = Path(m3u8_path) if m3u8_path else None
        self.fifo_path = Path(fifo_path) if fifo_path else Path(tempfile.mktemp())
        self.fifo_r = self.fifo_w = None
        self.stream_process = None
        self.reset()

    async def read(self, size: int = -1):
        if not self.fifo_r:
            fd_r = await asyncio.to_thread(os.open, self.fifo_path, os.O_RDONLY)
            fcntl.fcntl(fd_r, fcntl.F_SETPIPE_SZ, 1048576)
            self.fifo_r = os.fdopen(fd_r, "rb")
        data = await asyncio.to_thread(self.fifo_r.read, size)
        return data

    async def write(self, data):
        if not self.fifo_w:
            fd_w = await asyncio.to_thread(os.open, self.fifo_path, os.O_WRONLY)
            fcntl.fcntl(fd_w, fcntl.F_SETPIPE_SZ, 1048576)
            self.fifo_w = os.fdopen(fd_w, "wb")

        if self.sample_rate_r and self.sample_rate_w and self.sample_rate_w != self.sample_rate_r:
            data = resample_audio_bytes(data, self.sample_rate_w, self.sample_rate_r)

        size = await asyncio.to_thread(self.fifo_w.write, data)
        await asyncio.to_thread(self.fifo_w.flush)
        return size

    def reset(self, start_streaming=True):
        self.close()
        self.fifo_r = self.fifo_w = None

        if not self.fifo_path.exists():
            os.mkfifo(self.fifo_path)

        if self.stream_process:
            self.stream_process.terminate()
            self.stream_process = None

        if self.m3u8_path and start_streaming:
            self.m3u8_path.parent.mkdir(parents=True, exist_ok=True)
            self.m3u8_path.unlink(missing_ok=True)
            self.stream_to_m3u8(self.m3u8_path)

    def stream_to_m3u8(self, m3u8_path, sample_rate=16000):
        m3u8_path = Path(m3u8_path)
        m3u8_path.parent.mkdir(parents=True, exist_ok=True)
        self.ffmpeg_cmd = ["ffmpeg", "-f", "s16le"]
        self.ffmpeg_cmd.extend(["-v", "error"])
        self.ffmpeg_cmd.extend(["-ar", str(self.sample_rate_w)])
        self.ffmpeg_cmd.extend(["-ac", "1"])
        self.ffmpeg_cmd.extend(["-i", self.fifo_path.as_posix()])
        self.ffmpeg_cmd.extend(["-c:a", "libmp3lame"])
        self.ffmpeg_cmd.extend(["-b:a", "128k"])
        self.ffmpeg_cmd.extend(["-f", "hls"])
        self.ffmpeg_cmd.extend(["-hls_time", "1"])
        self.ffmpeg_cmd.extend(["-hls_list_size", "0"])
        self.ffmpeg_cmd.extend(["-hls_playlist_type", "event"])
        self.ffmpeg_cmd.extend([m3u8_path.as_posix()])
        logger.debug(" ".join(self.ffmpeg_cmd))
        self.stream_process = subprocess.Popen(self.ffmpeg_cmd)

    def close(self):
        if self.fifo_r:
            self.fifo_r.close()
        if self.fifo_w:
            self.fifo_w.close()


def resample_audio_bytes(audio_bytes, original_sample_rate=44100, target_sample_rate=16000):
    audio_data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    number_of_samples = int(len(audio_data) * (target_sample_rate / original_sample_rate))
    resampled_audio = resample(audio_data, number_of_samples)
    resampled_audio = (resampled_audio * 32768.0).astype(np.int16)
    resampled_audio_bytes = resampled_audio.tobytes()
    return resampled_audio_bytes


class FIFOTextIOStream:
    def __init__(self, fifo_path=None):
        if not fifo_path:
            fifo_path = tempfile.mktemp()
        self.fifo_path = Path(fifo_path)

        if not self.fifo_path.exists():
            os.mkfifo(self.fifo_path)

        self.fifo_r = self.fifo_w = None

    async def read(self, size: int = -1):
        if not self.fifo_r:
            self.fifo_r = await asyncio.to_thread(open, self.fifo_path, "r")
        data = await asyncio.to_thread(self.fifo_r.read, size)
        return data

    async def write(self, data):
        if self.fifo_w is None:
            self.fifo_w = await asyncio.to_thread(open, self.fifo_path, "w")
        if data:
            size = await asyncio.to_thread(self.fifo_w.write, data)
        else:
            size = 0
        await asyncio.to_thread(self.fifo_w.flush)
        return size

    def close(self):
        if self.fifo_w:
            self.fifo_w.close()
        if self.fifo_r:
            self.fifo_r.close()


if __name__ == "__main__":

    import time
    from multiprocessing import Process

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("test")
    logger.setLevel(logging.DEBUG)

    async def test_text():
        response = "你好" * 5

        text_stream = FIFOTextIOStream()
        logger.debug("text stream created")

        def read(text_fifo_path):
            with open(text_fifo_path, "r") as fifo:
                while True:
                    data = fifo.read(5)
                    if len(data) == 0:
                        break
                    logger.debug("read: " + data)
            logger.debug("text stream read finished")

        Process(target=read, args=(text_stream.fifo_path.as_posix(),)).start()

        for c in response:
            await text_stream.write(c)
            await asyncio.sleep(0.1)
        text_stream.close()
        logger.debug("text stream write closed")
        logger.debug("=" * 20)

        text_stream = FIFOTextIOStream()

        def write(text_fifo_path):
            with open(text_fifo_path, "w") as fifo:
                for c in response:
                    fifo.write(c)
                    fifo.flush()
                    time.sleep(0.1)
            logger.debug("text stream write finished")

        Process(target=write, args=(text_stream.fifo_path.as_posix(),)).start()
        while True:
            data = await text_stream.read(5)
            if not data:
                break
            logger.debug("read: " + data)

    async def test_audio():

        async def read_audio():
            while True:
                data = await stream.read(1024)
                if not data:
                    break
                print(len(data))

        task = asyncio.create_task(read_audio())
        stream = FIFOAudioIOStream(sample_rate_w=48000, sample_rate_r=16000)
        await stream.write(8192 * b"\x00")
        await stream.write(8192 * b"\x00")
        await stream.write(8192 * b"\x00")

        await task

    asyncio.run(test_text())
