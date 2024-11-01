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
    """
    A class for reading and writing audio data to a FIFO file.
    Attributes:
        mode (str): Mode for the FIFO file. Can be 'c', r', 'w', or 'rw'. Mode 'c' creates the fifo without managing it \
                If mode is 'w', then the stream will not be readable and "read from fifo" should be managed separately, see examples.
        fifo_path (str): Path to the FIFO file.
        sample_rate_r (int): Sample rate of the output audio stream. 
        sample_rate_w (int): Sample rate of the input audio stream. If sample_rate_w is specified, the input audio stream will be resampled from sample_rate_w to sample_rate_r.
    Examples:
        >>> stream = FIFOAudioIOStream(sample_rate_w=44100, sample_rate_r=16000)
        >>> await stream.write(audio_data)
        >>> data = await stream.read()

        Use "w" mode and manage fifo read separately:
        >>> stream = FIFOAudioIOStream(mode="w")
        >>> await stream.write(audio_data)
        >>> with open(stream.fifo_path, "r") as fifo:
        >>>     data = fifo.read()
    """

    def __init__(
        self,
        fifo_path: Optional[str | Path] = None,
        mode: str = "rw",
        sample_rate_r: int = None,
        sample_rate_w: int = None,
    ):
        self.sample_rate_r = sample_rate_r
        self.sample_rate_w = sample_rate_w

        # Set up the FIFO path
        if not fifo_path:
            fifo_path = tempfile.mktemp()
        self.fifo_path = Path(fifo_path)

        # Create the FIFO if it does not exist
        if not self.fifo_path.exists():
            os.mkfifo(self.fifo_path)

        assert mode in ["r", "w", "rw", "c"]
        if "r" in mode:
            fd_r = os.open(self.fifo_path, os.O_RDONLY | os.O_NONBLOCK)
            fcntl.fcntl(fd_r, fcntl.F_SETPIPE_SZ, 1048576)
            self.fifo_r = os.fdopen(fd_r, "rb")
            assert self.sample_rate_r, "sample_rate_r is required for 'r' mode"

        if "w" in mode:
            fd_w = os.open(self.fifo_path, os.O_WRONLY | os.O_NONBLOCK)
            fcntl.fcntl(fd_w, fcntl.F_SETPIPE_SZ, 1048576)
            self.fifo_w = os.fdopen(fd_w, "wb")
            assert self.sample_rate_w, "sample_rate_w is required for 'w' mode"

    async def read(self, size: int = -1):
        if not self.fifo_r:
            raise ValueError("Reading from fifo is not managed by this stream.")
        while True:
            data = await asyncio.to_thread(self.fifo_r.read, size)
            if not data:
                if self.fifo_r.closed:
                    break
                await asyncio.sleep(0.1)
                continue
            return data

    async def write(self, data):
        if not self.fifo_w:
            raise ValueError("Writing to fifo is not managed by this stream.")

        if self.sample_rate_r and self.sample_rate_w != self.sample_rate_r:
            data = resample_audio_bytes(data, self.sample_rate_w, self.sample_rate_r)

        await asyncio.to_thread(self.fifo_w.write, data)
        await asyncio.to_thread(self.fifo_w.flush)

    def stream_to_m3u8(self, m3u8_path, sample_rate=16000):
        m3u8_path = Path(m3u8_path)
        m3u8_path.parent.mkdir(parents=True, exist_ok=True)
        self.ffmpeg_cmd = ["ffmpeg", "-f", "s16le"]
        self.ffmpeg_cmd.extend(["-v", "error"])
        self.ffmpeg_cmd.extend(["-ar", str(self.sample_rate_w)])
        self.ffmpeg_cmd.extend(["-ac", "1"])
        self.ffmpeg_cmd.extend(["-i", self.fifo_path.as_posix()])
        self.ffmpeg_cmd.extend(["-c:a", "aac"])
        self.ffmpeg_cmd.extend(["-b:a", "192k"])
        self.ffmpeg_cmd.extend(["-ar", str(sample_rate)])
        self.ffmpeg_cmd.extend(["-f", "hls"])
        self.ffmpeg_cmd.extend(["-hls_time", "4"])
        self.ffmpeg_cmd.extend(["-hls_list_size", "0"])
        self.ffmpeg_cmd.extend(["-hls_playlist_type", "event"])
        self.ffmpeg_cmd.extend([m3u8_path.as_posix()])
        print(" ".join(self.ffmpeg_cmd))
        subprocess.Popen(self.ffmpeg_cmd)

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
    def __init__(self, fifo_path=None, mode="rw"):
        # Set up the FIFO path
        if not fifo_path:
            fifo_path = tempfile.mktemp()
        self.fifo_path = Path(fifo_path)

        # Create the FIFO if it does not exist
        if not self.fifo_path.exists():
            os.mkfifo(self.fifo_path)

        self.fifo_r_fd = self.fifo_w_fd = None
        if "w" in mode:
            try:
                self.fifo_w_fd = os.open(self.fifo_path, os.O_WRONLY | os.O_NONBLOCK)
            except OSError as e:
                if e.errno == errno.ENXIO:
                    self.fifo_w_fd = -1
                    logger.warning(f"Postponed opening of {self.fifo_path} for writing since no reader is opening it.")
                else:
                    raise
        if "r" in mode:
            self.fifo_r_fd = os.open(self.fifo_path, os.O_RDONLY | os.O_NONBLOCK)
            self.writer_never_opened = True
            self.buffer_read = b""

    async def read(self, size: int = -1):
        if not self.fifo_r_fd:
            raise ValueError("Stream is not readable")
        while True:
            try:
                data = await asyncio.to_thread(os.read, self.fifo_r_fd, size)
                if len(data) == 0:
                    # EOF or no writer is opening
                    if self.writer_never_opened:
                        await asyncio.sleep(0.1)
                        continue
                    else:
                        break
                self.buffer_read += data
                text, bytes_decoded = "", 0
                try:
                    text = self.buffer_read.decode("u8")
                    self.buffer_read = b""
                except UnicodeDecodeError as e:
                    # Partial character, keep self.buffer_read and try again
                    bytes_decoded = e.start
                    text = self.buffer_read[:bytes_decoded].decode("u8")
                    self.buffer_read = self.buffer_read[bytes_decoded:]
                if text:
                    return text
            except OSError as e:
                if e.errno == errno.EAGAIN:
                    # no data available yet, but there is writer opening
                    self.writer_never_opened = False
                    await asyncio.sleep(0.1)
                    continue
                raise e

    async def write(self, data):
        if not self.fifo_w_fd:
            raise ValueError("Stream is not writable")
        if self.fifo_w_fd == -1:
            while True:
                try:
                    self.fifo_w_fd = os.open(self.fifo_path, os.O_WRONLY | os.O_NONBLOCK)
                    break
                except OSError as e:
                    if e.errno == errno.ENXIO:
                        await asyncio.sleep(0.1)
                        continue
                    else:
                        raise
        await asyncio.to_thread(os.write, self.fifo_w_fd, data.encode("u8"))

    def close(self):
        if self.fifo_w_fd and self.fifo_w_fd != -1:
            os.close(self.fifo_w_fd)
        if self.fifo_r_fd:
            os.close(self.fifo_r_fd)


if __name__ == "__main__":

    from multiprocessing import Process

    async def main_text():
        text_stream = FIFOTextIOStream(mode="w")

        response = "你好" * 10

        import time

        def test(text_fifo_path):

            with open(text_fifo_path, "r") as fifo:
                while True:
                    data = fifo.read(5)
                    if len(data) == 0:
                        break
                    print(data)
            print("text stream read finished")

        Process(target=test, args=(text_stream.fifo_path.as_posix(),)).start()

        for c in response:
            await text_stream.write(c)
            time.sleep(0.1)
        text_stream.close()
        print("text stream closed")

    async def main_audio():

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

    asyncio.run(main_text())
