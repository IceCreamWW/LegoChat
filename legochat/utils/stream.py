import asyncio
import fcntl
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from scipy.signal import resample


class AudioInputStream:
    pass


class AudioOutputStream:
    pass


class FIFOAudioIOStream(AudioInputStream, AudioOutputStream):
    def __init__(self, fifo_path=None, mode="rw", sample_rate_in=16000, sample_rate_out=16000):
        self.sample_rate_in = sample_rate_in
        self.sample_rate_out = sample_rate_out

        # Set up the FIFO path
        if not fifo_path:
            fifo_path = tempfile.mktemp()
        self.fifo_path = Path(fifo_path)

        # Create the FIFO if it does not exist
        if not self.fifo_path.exists():
            os.mkfifo(self.fifo_path)

        if "w" in mode:
            rw_flag = os.O_RDWR
        elif "r" in mode:
            rw_flag = os.O_RDONLY
        else:
            raise ValueError(f"Invalid mode {mode}")

        fd = os.open(self.fifo_path, rw_flag | os.O_NONBLOCK)
        fcntl.fcntl(fd, fcntl.F_SETPIPE_SZ, 1048576)

        # Open separate file descriptors for reading and writing
        self.fifo_w = open(fd, "wb") if "w" in mode else None
        self.fifo_r = open(fd, "rb") if "r" in mode else None

    async def read(self, size: int = -1):
        if not self.fifo_r:
            raise ValueError("Stream is not readable")
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
            raise ValueError("Stream is not writable")
        if self.sample_rate_in != self.sample_rate_out:
            data = resample_audio_bytes(data, self.sample_rate_in, self.sample_rate_out)

        await asyncio.to_thread(self.fifo_w.write, data)
        await asyncio.to_thread(self.fifo_w.flush)  # Ensure the data is flushed

    def close(self):
        """Close both read and write file descriptors."""
        self.fifo_r.close()
        self.fifo_w.close()


class M3U8AudioOutputStream(AudioOutputStream):
    def __init__(self, m3u8_path, fifo_path=None):
        self.m3u8_path = m3u8_path

        self.fifo_path = tempfile.mktemp() if not fifo_path else fifo_path
        self.fifo_path = Path(self.fifo_path)
        self.fifo = None
        self.ffmpeg_cmd = ["ffmpeg", "-f", "s16le"]
        self.ffmpeg_cmd.extend(["-ar", "16k"])
        self.ffmpeg_cmd.extend(["-ac", "1"])
        self.ffmpeg_cmd.extend(["-i", self.fifo_path.as_posix()])
        self.ffmpeg_cmd.extend(["-c:a", "aac"])
        self.ffmpeg_cmd.extend(["-b:a", "192k"])
        self.ffmpeg_cmd.extend(["-f", "hls"])
        self.ffmpeg_cmd.extend(["-hls_time", "4"])
        self.ffmpeg_cmd.extend(["-hls_list_size", "0"])
        self.ffmpeg_cmd.extend(["-hls_playlist_type", "event"])
        self.ffmpeg_cmd.extend([m3u8_path.as_posix()])

    async def write(self, data):
        if not self.fifo_path.exists() or self.fifo is None or self.fifo.closed:
            os.mkfifo(self.fifo_path)
            self.fifo = open(self.fifo_path, "wb")
            subprocess.Popen(self.ffmpeg_cmd)
        await asyncio.to_thread(self.fifo.write, data)


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

        if "w" in mode:
            rw_flag = os.O_RDWR
        elif "r" in mode:
            rw_flag = os.O_RDONLY
        else:
            raise ValueError(f"Invalid mode {mode}")

        fd = os.open(self.fifo_path, rw_flag | os.O_NONBLOCK)
        # Open separate file descriptors for reading and writing
        self.fifo_w = os.fdopen(fd, "w") if "w" in mode else None
        self.fifo_r = os.fdopen(fd, "r") if "r" in mode else None

    async def read(self, size: int = -1):
        if not self.fifo_r:
            raise ValueError("Stream is not readable")
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
            raise ValueError("Stream is not writable")
        await asyncio.to_thread(self.fifo_w.write, data)
        await asyncio.to_thread(self.fifo_w.flush)  # Ensure the data is flushed

    def close(self):
        """Close both read and write file descriptors."""
        self.fifo_r.close()
        self.fifo_w.close()


if __name__ == "__main__":

    from multiprocessing import Process

    async def main_text():
        fifo_path = Path("/tmp/my_fifo")
        fifo_path.unlink(missing_ok=True)
        os.mkfifo(fifo_path)
        fd = os.open(fifo_path, os.O_RDONLY | os.O_NONBLOCK)
        fifo_r = os.fdopen(fd, "r")

        def test(text_fifo_path):
            import time

            print("TESTING")
            response = "测试用这个回复" * 100
            with open(text_fifo_path, "w") as fifo:
                for c in response:
                    print("writing", c)
                    fifo.write(c)
                    fifo.flush()
                    time.sleep(0.5)

        asyncio.create_task(asyncio.to_thread(test, fifo_path.as_posix()))

        while True:
            data = await asyncio.to_thread(fifo_r.read, 1)
            if not data:
                if fifo_r.closed:
                    break
                await asyncio.sleep(0.1)
                continue
            print("received data", data)

    async def main_audio():

        async def read_audio():
            while True:
                data = await stream.read(1024)
                if not data:
                    break
                print(len(data))

        task = asyncio.create_task(read_audio())
        stream = FIFOAudioIOStream(sample_rate_in=48000, sample_rate_out=16000)
        await stream.write(8192 * b"\x00")
        await stream.write(8192 * b"\x00")
        await stream.write(8192 * b"\x00")

        await task

    # asyncio.run(main_text())
    FIFOTest = FIFOTextIOStream(mode="w")
