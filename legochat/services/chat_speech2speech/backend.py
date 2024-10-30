import argparse
import asyncio
import tempfile
from multiprocessing import Queue
from pathlib import Path
from typing import Optional
from uuid import uuid4

import yaml
from legochat.components import VAD, Chatbot, Speech2Text, Text2Speech
from legochat.services.service import Service
from legochat.utils.event import EventBus, EventEnum
from legochat.utils.stream import (AudioInputStream, AudioOutputStream,
                                   TextOutputStream)


class ChatSpeech2Speech(Service):
    def __init__(self, config):
        super().__init__(config)

    @property
    def required_components(self):
        return ["vad", "speech2text", "chatbot", "text2speech"]

    async def start_session(self, audio_stream):
        session_id = uuid4().hex
        session = Session(
            self,
            session_id,
            audio_stream,
        )
        self.sessions.append(session)
        asyncio.create_task(session.run())
        return session_id


class Session:
    def __init__(
        self,
        service,
        session_id,
        user_audio_input_stream: AudioInputStream,
        agent_audio_output_stream: AudioOutputStream,
        voiced_seconds_to_interrupt: float = -1,
        unvoiced_seconds_to_eot: float = -1,
        allow_vad_interrupt: bool = True,
        allow_vad_eot: bool = True,
    ):
        for component in service.required_components:
            setattr(self, component, service.components[component])

        self.workspace = Path(tempfile.TemporaryDirectory().name)
        self.workspace.mkdir(parents=True, exist_ok=True)

        self.data = b""
        self.service = service
        self.session_id = session_id
        self.sample_rate = 16000

        self.vad_states = None
        self.voiced_segments = []
        self.transcripts = []
        self.chat_messages = self.chatbot.system_prompt
        self.chatbot_controller = None
        self.text2speech_controller = None

        self.user_audio_input_stream = user_audio_input_stream
        self.agent_audio_output_stream = agent_audio_output_stream
        self.user_text_output_stream = user_text_output_stream
        self.agent_text_output_stream = agent_text_output_stream

        self.voiced_bytes_to_interrupt = int(voiced_seconds_to_interrupt * self.sample_rate) * 2
        self.unvoiced_bytes_to_eot = int(unvoiced_seconds_to_eot * self.sample_rate) * 2

        self.allow_vad_interrupt = allow_vad_interrupt
        self.allow_vad_eot = allow_vad_eot

        self.event_bus = EventBus()
        self.agent_can_speak = True
        self.event_bus.on(EventEnum.INTERRUPT, self.on_interrupt)
        self.event_bus.on(EventEnum.END_OF_TURN, self.on_end_of_turn)

    async def run(self):
        while True:
            chunk = await self.audio_stream.read()
            if not chunk:
                break
            self.process_chunk(chunk)

    async def on_interrupt(self, sender="user"):
        if not self.allow_vad_interrupt and sender == "vad":
            return
        self.agent_can_speak = False
        if self.chatbot_controller:
            self.chatbot_controller.put("interrupt")
        if self.text2speech_controller:
            self.text2speech_controller.put("interrupt")

    async def on_end_of_turn(self, sender="user"):
        if not self.allow_vad_eot and sender == "vad":
            return
        self.agent_can_speak = True
        await self.agent_speak()

    async def agent_speak(self):
        transcript = self.transcript
        self.voiced_segments = []
        self.transcripts = []
        self.chatbot_controller = Queue()
        self.text2speech_controller = Queue()

        self.chat_messages = self.chatbot.add_user_message(self.chat_messages, transcript)
        chatbot_text_output_stream = TextIOStream()
        asyncio.create_task(
            self.chatbot.process(
                chat_messages=self.chat_messages,
                text_fifo_path=chatbot_text_output_stream.text_fifo_path.as_posix(),
                control_pipe=self.chatbot_controller,
            )
        )

        text2speech_text_input_stream = PipeTextIOStream()
        asyncio.create_task(
            self.text2speech.process(
                text_fifo_path=text2speech_text_input_stream.text_fifo_path.as_posix(),
                audio_fifo_path=self.agent_audio_output_stream.audio_fifo_path.as_posix(),
                control_pipe=self.text2speech_controller,
            )
        )

        chatbot_response = ""
        chat_messages_partial = None
        while True:
            message_partial = await chatbot_text_output_stream.read()
            if not message_partial:
                break
            chatbot_response += message_partial
            chat_messages_partial = self.chatbot.add_agent_message(self.chat_messages, chatbot_response)
            # await self.event_bus.trigger_event(EventEnum.UPDATE_CHAT_MESSAGES, chat_messages_partial)

        if chat_messages_partial:
            self.chat_messages = chat_messages_partial

    async def process_chunk(self, chunk):
        vad_results, self.vad_states = await self.vad.process(samples=chunk, prev_states=self.vad_states)
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

        if (
            self.unvoiced_bytes_to_eot > 0
            and len(self.data) - self.voiced_segments[-1]["end"] > self.unvoiced_bytes_to_eot
        ):
            await self.event_bus.trigger_event(EventEnum.END_OF_TURN, sender="vad")
        elif (
            self.voiced_bytes_to_interrupt > 0
            and len(self.data) == self.voiced_segments[-1]["end"]
            and self.voiced_segments[-1]["end"] - self.voiced_segments[-1]["start"] > self.voiced_bytes_to_interrupt
        ):
            await self.event_bus.trigger_event(EventEnum.INTERRUPT, sender="vad")

        start, end = self.voiced_segments[-1]["start"], self.voiced_segments[-1]["end"]
        text = await self.speech2text.process(data)
        await self.update_transcript(start, end, text)

    @property
    def transcript(self):
        return "".join([transcript[2] for transcript in self.transcripts])

    @property
    def speech(self):
        data = b""
        for segment in voiced_segments:
            data += self.data[segment["start"] : segment["end"]]

    async def update_transcript(self, start, end, text):
        outdated_transcripts = []
        for start_, end_, _ in self.transcripts:
            if start >= start_ and end <= end_:
                return None
            elif start <= start_ and end >= end_:
                outdated_transcripts.append((start, end, text))
            elif start_ < start < end_ or start_ < end < end_:
                raise ValueError("Overlapping transcripts")
        for outdated_transcript in outdated_transcripts:
            self.transcripts.remove(outdated_transcript)
        self.transcripts.append((start, end, text))
        self.transcripts.sort(key=lambda x: x[0])
        # self.event_bus.trigger_event(EventEnum.UPDATE_TRANSCRIPT, self.transcript)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
