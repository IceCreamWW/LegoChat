import asyncio
import os
import subprocess
import tempfile
from pathlib import Path


class AudioInputStream:
    pass


class AudioOutputStream:
    pass


class FIFOAudioIOStream(AudioInputStream, AudioOutputStream):
    def __init__(self, fifo_path=None):
        # Set up the FIFO path
        if not fifo_path:
            fifo_path = tempfile.mktemp()
        self.fifo_path = Path(fifo_path)

        # Create the FIFO if it does not exist
        if not self.fifo_path.exists():
            os.mkfifo(self.fifo_path)

        # Open separate file descriptors for reading and writing
        self.fifo_r = open(self.fifo_path, "rb")
        self.fifo_w = open(self.fifo_path, "wb")

    async def read(self, size: int = -1):
        """Asynchronously read from the FIFO."""
        return await asyncio.to_thread(self.fifo_r.read, size)

    async def write(self, data):
        """Asynchronously write to the FIFO."""
        await asyncio.to_thread(self.fifo_w.write, data)
        await asyncio.to_thread(self.fifo_w.flush)  # Ensure the data is flushed

    def close(self):
        """Close both read and write file descriptors."""
        self.fifo_r.close()
        self.fifo_w.close()


class M3U8AudioOutputStream(AudioOutputStream):
    def __init__(self, m3u8_path, fifo_path=None):
        self.m3u8_path = m3u8_path

        if not fifo_path:
            self.fifo_path = tempfile.mktemp()
        self.fifo_path = Path(self.fifo_path)
        self.fifo = None
        self.ffmpeg_cmd = ["ffmpeg", "-f", "s16le"]
        self.ffmpeg_cmd.extend(["-ar", "16k"])
        self.ffmpeg_cmd.extend(["-ac", "1"])
        self.ffmpeg_cmd.extend(["-i", fifo_path.as_posix()])
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
