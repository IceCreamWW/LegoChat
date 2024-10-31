import argparse
import asyncio
import logging
import tempfile
from multiprocessing import Pipe
from pathlib import Path
from typing import Optional
from uuid import uuid4

import yaml
from legochat.services.service import Service
from legochat.utils.event import EventBus, EventEnum
from legochat.utils.stream import (AudioInputStream, AudioOutputStream,
                                   FIFOTextIOStream)


class ChatSpeech2Speech(Service):
    def __init__(self, config):
        super().__init__(config)
        self.sessions = {}

    @property
    def required_components(self):
        return ["vad", "speech2text", "chatbot", "text2speech"]

    async def start_session(self, **session_kwargs):
        session = ChatSpeech2SpeechSession(self, **session_kwargs)
        self.sessions[session.session_id] = session
        await session.run()


class ChatSpeech2SpeechSession:
    def __init__(
        self,
        service,
        session_id,
        user_audio_input_stream: AudioInputStream,
        agent_audio_output_stream: AudioOutputStream,
        workspace: Optional[Path] = None,
        voiced_seconds_to_interrupt: float = -1,
        unvoiced_seconds_to_eot: float = -1,
        allow_vad_interrupt: bool = True,
        allow_vad_eot: bool = True,
    ):
        for component in service.required_components:
            setattr(self, component, service.components[component])

        self.workspace = Path(tempfile.TemporaryDirectory().name) if not workspace else Path(workspace)
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

        self.voiced_bytes_to_interrupt = int(voiced_seconds_to_interrupt * self.sample_rate) * 2
        self.unvoiced_bytes_to_eot = int(unvoiced_seconds_to_eot * self.sample_rate) * 2

        self.allow_vad_interrupt = allow_vad_interrupt
        self.allow_vad_eot = allow_vad_eot

        self.event_bus = EventBus()
        self.agent_can_speak = False
        self.user_is_speaking = False
        self.event_bus.on(EventEnum.RECEIVE_ADUIO_CHUNK, self.on_receive_audio_chunk)
        self.event_bus.on(EventEnum.UPDATE_VOICED_SEGMENTS, self.on_update_voiced_segments)
        self.event_bus.on(EventEnum.INTERRUPT, self.on_interrupt)
        self.event_bus.on(EventEnum.END_OF_TURN, self.on_end_of_turn)

        self.last_vad_end = 0
        self.last_speech2text_end = 0

    async def run(self):
        asyncio.create_task(self.detect_speech())
        try:
            while True:
                chunk = await self.user_audio_input_stream.read(2048)
                if not chunk:
                    break
                await self.event_bus.emit(EventEnum.RECEIVE_ADUIO_CHUNK, chunk)
        except Exception as e:
            traceback.print_exc()

    async def on_receive_audio_chunk(self, chunk: bytes):
        self.data += chunk
        await self.detect_speech()

    async def on_update_voiced_segments(self):
        asyncio.create_task(self.transcribe())

    async def on_interrupt(self, sender="user"):
        if not self.allow_vad_interrupt and sender == "vad":
            return
        self.agent_can_speak = False
        if self.chatbot_controller:
            self.chatbot_controller.send("interrupt")
        if self.text2speech_controller:
            self.text2speech_controller.send("interrupt")

    def on_end_of_turn(self, sender="user"):
        print(f"end of turn received from {sender}")
        if not self.allow_vad_eot and sender == "vad":
            return
        if self.agent_can_speak:
            return
        self.agent_can_speak = True
        asyncio.create_task(self.agent_speak())

    async def detect_speech(self):
        chunk = self.data[self.last_vad_end :]
        if not chunk:
            return

        self.last_vad_end = len(self.data)
        vad_results, self.vad_states = await asyncio.to_thread(
            self.vad.process, samples=chunk, prev_states=self.vad_states
        )
        voice_segments_updated = False
        for result in vad_results:
            if "start" in result:
                self.user_is_speaking = True
                if self.voiced_segments and self.voiced_segments[-1]["end"] >= result["start"] * 2:
                    continue
                self.voiced_segments.append({"start": result["start"] * 2})
            elif "end" in result:
                self.user_is_speaking = False
                self.voiced_segments[-1]["end"] = result["end"] * 2
            voice_segments_updated = True

        if self.user_is_speaking:
            self.voiced_segments[-1]["end"] = self.last_vad_end
            voice_segments_updated = True

        if voice_segments_updated:
            await self.event_bus.emit(EventEnum.UPDATE_VOICED_SEGMENTS)

        if (
            self.unvoiced_bytes_to_eot > 0
            and self.last_vad_end - self.voiced_segments[-1]["end"] > self.unvoiced_bytes_to_eot
            and not self.agent_can_speak
        ):
            await self.event_bus.emit(EventEnum.END_OF_TURN, sender="vad")
        elif (
            self.voiced_bytes_to_interrupt > 0
            and self.last_vad_end == self.voiced_segments[-1]["end"]
            and self.voiced_segments[-1]["end"] - self.voiced_segments[-1]["start"] > self.voiced_bytes_to_interrupt
            and self.agent_can_speak
        ):
            await self.event_bus.emit(EventEnum.INTERRUPT, sender="vad")

    async def transcribe(self):
        start, end = self.voiced_segments[0]["start"], self.voiced_segments[-1]["end"]
        text, _ = await asyncio.to_thread(self.speech2text.process, samples=self.speech)
        self.last_speech2text_end = end
        self.update_transcript(start, end, text)
        logging.info(f"Transcription from {start} to {end}: {text}")

    @property
    def transcript(self):
        return "".join([transcript[2] for transcript in self.transcripts])

    @property
    def speech(self):
        data = b""
        for segment in self.voiced_segments:
            data += self.data[segment["start"] : segment["end"]]
        return data

    def update_transcript(self, start, end, text):
        outdated_transcripts = []
        for start_, end_, text_ in self.transcripts:
            if start >= start_ and end <= end_:
                return None
            elif start <= start_ and end >= end_:
                outdated_transcripts.append((start_, end_, text_))
            elif start_ < start < end_ or start_ < end < end_:
                raise ValueError("Overlapping transcripts")
        for outdated_transcript in outdated_transcripts:
            self.transcripts.remove(outdated_transcript)
        self.transcripts.append((start, end, text))
        self.transcripts.sort(key=lambda x: x[0])

    async def agent_speak(self):
        transcript = self.transcript
        self.voiced_segments = []
        self.transcripts = []
        self.chatbot_controller, chatbot_controller_child = Pipe()
        self.text2speech_controller, text2speech_controller_child = Pipe()

        self.chat_messages = self.chatbot.add_user_message(self.chat_messages, transcript)
        chatbot_text_output_stream = FIFOTextIOStream(mode="r")

        asyncio.create_task(
            asyncio.to_thread(
                self.chatbot.process,
                messages=self.chat_messages,
                text_fifo_path=chatbot_text_output_stream.fifo_path.as_posix(),
                control_pipe=chatbot_controller_child,
            )
        )

        # asyncio.create_task(asyncio.to_thread(test, chatbot_text_output_stream.fifo_path.as_posix()))
        while True:
            print("waiting for chatbot response")
            data = await chatbot_text_output_stream.read(1)
            if not data:
                break
            print("received ", data)
        return

        text2speech_text_input_stream = FIFOTextIOStream(mode="rw")
        asyncio.create_task(
            asyncio.to_thread(
                self.text2speech.process,
                text_fifo_path=text2speech_text_input_stream.fifo_path.as_posix(),
                audio_fifo_path=self.agent_audio_output_stream.fifo_path.as_posix(),
                control_pipe=text2speech_controller_child,
            )
        )

        chatbot_response = ""
        chat_messages = self.chat_messages
        while True:
            print("receving")
            message_partial = await chatbot_text_output_stream.read(5)
            print("read message partial", message_partial)
            if not message_partial:
                break
            await text2speech_text_input_stream.write(message_partial)
            chatbot_response += message_partial
            self.chat_messages = self.chatbot.add_agent_message(chat_messages, chatbot_response)
            # await self.event_bus.emit(EventEnum.UPDATE_CHAT_MESSAGES, chat_messages_partial)


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
