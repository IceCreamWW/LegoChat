import asyncio
import errno
import logging
import os
import tempfile
import traceback
from multiprocessing import Pipe
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from legochat.services.service import Service
from legochat.utils.event import EventBus, EventEnum
from legochat.utils.stream import (AudioInputStream, AudioOutputStream,
                                   FIFOTextIOStream)
from repair_report import RepairReportBot

logger = logging.getLogger("legochat")
env_file = Path(__file__).parent / ".env"
if env_file.exists():
    load_dotenv(env_file)

SPEECH2TEXT_MIN_SECONDS = float(os.getenv("SPEECH2TEXT_MIN_SECONDS", "0.2"))
SPEECH2TEXT_MAX_SECONDS = float(os.getenv("SPEECH2TEXT_MAX_SECONDS", "5.0"))
SPEECH2TEXT_MIN_INTERVAL_SECONDS = float(os.getenv("SPEECH2TEXT_MIN_INTERVAL_SECONDS", "0.2"))
VOICED_SECONDS_TO_INTERRUPT = float(os.getenv("VOICED_SECONDS_TO_INTERRUPT", "0.3"))
UNVOICED_SECONDS_TO_EOT = float(os.getenv("UNVOICED_SECONDS_TO_EOT", "0.6"))
CHATBOT_MAX_MESSAGES = int(os.getenv("CHATBOT_MAX_MESSAGES", "50"))


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
        asyncio.create_task(self.heartbeat(session.session_id))
        await session.run()

    async def heartbeat(self, session_id):
        while True:
            self.sessions[session_id].is_alive = False
            await asyncio.sleep(10)
            if not self.sessions[session_id].is_alive:
                break
        logger.info(f"Session {session_id} ended")
        del self.sessions[session_id]


class ChatSpeech2SpeechSession:
    def __init__(
        self,
        service,
        session_id,
        user_audio_input_stream: AudioInputStream,
        agent_audio_output_stream: AudioOutputStream,
        workspace: Optional[Path] = None,
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

        self.repair_report_bot = RepairReportBot()

        self.vad_states = None
        self.voiced_segments = []
        self.transcripts = []
        self.chatbot.system_prompt = repair_report_bot.system_prompt
        self.chat_messages = self.chatbot.system_prompt
        self.text2speech_controller = None

        self.user_audio_input_stream = user_audio_input_stream
        self.agent_audio_output_stream = agent_audio_output_stream

        self.allow_vad_interrupt = allow_vad_interrupt
        self.allow_vad_eot = allow_vad_eot

        self.speech2text_min_bytes = int(SPEECH2TEXT_MIN_SECONDS * self.sample_rate * 2)
        self.speech2text_max_bytes = int(SPEECH2TEXT_MAX_SECONDS * self.sample_rate * 2)
        self.speech2text_min_interval_bytes = int(SPEECH2TEXT_MIN_INTERVAL_SECONDS * self.sample_rate * 2)
        self.speech2text_offset_bytes = 0
        self.voiced_bytes_to_interrupt = int(VOICED_SECONDS_TO_INTERRUPT * self.sample_rate) * 2
        self.unvoiced_bytes_to_eot = int(UNVOICED_SECONDS_TO_EOT * self.sample_rate) * 2

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
        logger.debug(f"interrupt received from {sender}")
        if not self.allow_vad_interrupt and sender == "vad":
            return
        self.agent_can_speak = False
        if self.text2speech_controller:
            # the text2speech_controller might already be closed
            try:
                self.text2speech_controller.send("interrupt")
                self.text2speech_controller.close()
            except OSError as e:
                if e.errno != errno.EPIPE:
                    raise e
                self.text2speech_controller = None

    def on_end_of_turn(self, sender="user"):
        logger.debug(f"end of turn received from {sender}")
        if not self.allow_vad_eot and sender == "vad":
            return
        if self.agent_can_speak:
            return
        transcript = self.transcript
        if not transcript.strip():
            return

        self.clear_transcript()
        self.agent_audio_output_stream.reset()
        self.agent_can_speak = True
        asyncio.create_task(self.agent_speak(transcript))

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
                    self.voiced_segments.pop()
                self.voiced_segments.append({"start": result["start"] * 2, "eos": False})
            if "end" in result:
                self.user_is_speaking = False
                self.voiced_segments[-1]["end"] = result["end"] * 2
                self.voiced_segments[-1]["eos"] = True
            voice_segments_updated = True

        if self.user_is_speaking:
            self.voiced_segments[-1]["end"] = self.last_vad_end
            self.voiced_segments[-1]["eos"] = False
            voice_segments_updated = True

        if voice_segments_updated:
            await self.event_bus.emit(EventEnum.UPDATE_VOICED_SEGMENTS)

        if (
            not self.agent_can_speak
            and self.voiced_segments
            and self.last_vad_end - self.voiced_segments[-1]["end"] > self.unvoiced_bytes_to_eot
        ):
            await self.event_bus.emit(EventEnum.END_OF_TURN, sender="vad")
        elif (
            self.agent_can_speak
            and self.voiced_segments
            and self.last_vad_end == self.voiced_segments[-1]["end"]
            and self.voiced_segments[-1]["end"] - self.voiced_segments[-1]["start"] > self.voiced_bytes_to_interrupt
        ):
            await self.event_bus.emit(EventEnum.INTERRUPT, sender="vad")

    async def transcribe(self):
        """Optimization in this functions applies to simulating online
        speech2text with offline speech2text model."""
        end, eos = self.voiced_segments[-1]["end"], self.voiced_segments[-1]["eos"]
        if end <= self.last_speech2text_end:
            return
        if not eos:
            if end - self.last_speech2text_end < self.speech2text_min_interval_bytes:
                return
            if len(self.speech) < self.speech2text_min_bytes:
                return
        if len(self.speech) > self.speech2text_max_bytes:
            if self.last_speech2text_end < self.voiced_segments[-1]["start"]:
                self.speech2text_offset_bytes = self.voiced_segments[-1]["start"]
        start, end = self.speech2text_offset_bytes, self.voiced_segments[-1]["end"]
        self.last_speech2text_end = max(end, self.last_speech2text_end)
        text, _ = await asyncio.to_thread(self.speech2text.process, samples=self.speech)
        logger.debug(f"text from {start} to {end}: {text}")
        self.update_transcript(start, end, text)

    @property
    def transcript(self):
        return "".join([transcript[2] for transcript in self.transcripts])

    @property
    def speech(self):
        data = b""
        for segment in self.voiced_segments:
            if segment["start"] >= self.speech2text_offset_bytes:
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

    def clear_transcript(self):
        if not self.voiced_segments:
            return
        last_voiced_segment = self.voiced_segments[-1]
        self.voiced_segments.clear()
        self.transcripts.clear()

        # handle manual EOT when VAD has not detect eos yet
        if not last_voiced_segment["eos"]:
            self.voiced_segments.append(
                {"start": last_voiced_segment["end"], "end": last_voiced_segment["end"], "eos": False}
            )

    async def agent_speak(self, transcript):
        self.text2speech_controller, text2speech_controller_child = Pipe()

        self.chat_messages = self.chatbot.add_user_message(self.chat_messages, transcript)

        if len(self.chat_messages) > CHATBOT_MAX_MESSAGES:
            self.chat_messages = [self.chat_messages[0]] + self.chat_messages[-CHATBOT_MAX_MESSAGES:]

        response = asyncio.to_thread(
            self.chatbot.process,
            messages=self.chat_messages,
        )

        text2speech_source_stream = FIFOTextIOStream()
        asyncio.create_task(
            asyncio.to_thread(
                self.text2speech.process,
                text_fifo_path=text2speech_source_stream.fifo_path.as_posix(),
                audio_fifo_path=self.agent_audio_output_stream.fifo_path.as_posix(),
                control_pipe=text2speech_controller_child,
            )
        )

        try:
            await text2speech_source_stream.write(response)  # make sure the fifo is alive
            self.repair_report_bot.add_assistant_message(dict(role="assistant", content=response))
            next_question = self.repair_report_bot.next_question()
            if next_question is None:
                self.agent_can_speak = False
                self.chat_messages = self.chatbot.add_agent_message(chat_messages, self.repair_report_bot.to_dict())
            else:
                self.chat_messages = self.chatbot.add_agent_message(chat_messages, next_question)
                await text2speech_source_stream.write(next_question)
                text2speech_source_stream.close()
        except Exception as e:
            self.agent_can_speak = False
            if isinstance(e, OSError):
                if e.errno != errno.EPIPE:
                    raise e
            logger.error(e)
