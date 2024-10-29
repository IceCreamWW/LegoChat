import argparse
import asyncio
from pathlib import Path

import yaml
from legochat.components import VAD, Chatbot, Speech2Text, Text2Speech
from legochat.services.service import Service


class ChatSpeech2Speech(Service):
    def __init__(self, config):
        self.components = self.build_components(config["components"])
        self.sessions = []

    async def start_session(self, audio_stream):
        session = Session(self)
        self.sessions.append(session)
        while True:
            chunk = await audio_stream.read()
            if not chunk:
                break
            session.process_chunk(chunk)


class Session:
    def __init__(self, service):
        self.data = b""
        self.service = service

        self.vad_states = None
        self.voiced_segments = []
        self.transcripts = []

    async def process_chunk(self, chunk):
        vad_results, self.vad_states = await self.service.components["vad"].process(chunk, self.vad_states)
        for result in vad_results:
            if "start" in result:
                if self.voiced_segments and self.voiced_segments[-1]["end"] >= result["start"] * 2:
                    continue
                self.voiced_segments.append({"start": result["start"] * 2})
            elif "end" in result:
                self.voiced_segments[-1]["end"] = result["end"] * 2
        self.data += chunk
        if self.voiced_segments and "end" not in self.voiced_segments[-1]:
            self.voiced_segments[-1]["end"] = len(self.data)

        transcript = await self.speech2text()
        self.update_transcripts(transcript)

    async def speech2text(self):
        segments = [segment for segment in self.voiced_segments if segment["start"] > self.offset]
        data = b""
        for segment in segments:
            data += self.data[segment["start"] : segment["end"]]
        start, end = segments[0]["start"], segments[-1]["end"]
        text = await self.service.components["speech2text"].process(data, None)
        return start, end, text

    def update_transcript(self, transcript):
        start_, end_, text_ = transcript
        outdated_transcripts = []
        for start, end, text in self.transcripts:
            if start_ >= start and end_ <= end:
                return None
            elif start_ <= start and end_ >= end:
                outdated_transcripts.append((start, end, text))
            elif start <= start_ <= end or start <= end_ <= end:
                raise ValueError("Overlapping transcripts")
        for outdated_transcript in outdated_transcripts:
            self.transcripts.remove(outdated_transcript)
        self.transcripts.append(transcript)
        self.transcripts.sort(key=lambda x: x[0])

    def reset_audio(self):
        self.vad_states = None
        self.data = b""
        self.voiced_segments = []
        self.transcripts = []


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())

    chat = ChatSpeech2Speech(vad, speech2text, chatbot, text2speech)
    asyncio.run(chat.run())
