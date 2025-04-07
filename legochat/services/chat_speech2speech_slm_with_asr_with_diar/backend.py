from typing import Dict, List
import asyncio
import errno
import logging
import os
import tempfile
import traceback
from multiprocessing import Pipe
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv
from legochat.services.service import Service
from legochat.utils.event import EventBus, EventEnum
from legochat.utils.stream import AudioInputStream, AudioOutputStream, FIFOTextIOStream
import uuid
import hashlib

logger = logging.getLogger("legochat")
env_file = Path(__file__).parent / ".env"
if env_file.exists():
    load_dotenv(env_file)

SPEECH2TEXT_MIN_SECONDS = float(os.getenv("SPEECH2TEXT_MIN_SECONDS", "0.2"))
SPEECH2TEXT_MAX_SECONDS = float(os.getenv("SPEECH2TEXT_MAX_SECONDS", "5.0"))
SPEECH2TEXT_MIN_INTERVAL_SECONDS = float(
    os.getenv("SPEECH2TEXT_MIN_INTERVAL_SECONDS", "0.2")
)
VOICED_SECONDS_TO_INTERRUPT = float(os.getenv("VOICED_SECONDS_TO_INTERRUPT", "0.3"))
UNVOICED_SECONDS_TO_EOT = float(os.getenv("UNVOICED_SECONDS_TO_EOT", "0.6"))
CHATBOT_MAX_MESSAGES = int(os.getenv("CHATBOT_MAX_MESSAGES", "20"))
VAD_LEFT_SILENCE_SECONDS = 0.5


class ChatSpeech2Speech(Service):
    def __init__(self, config):
        super().__init__(config)
        self.sessions = {}

    @property
    def required_components(self):
        return ["vad", "speech2text", "chatbot_slm", "text2speech", "diarization"]

    async def start_session(self, **session_kwargs):
        try:
            session = ChatSpeech2SpeechSession(self, **session_kwargs)
            self.sessions[session.session_id] = session
            asyncio.create_task(self.heartbeat(session.session_id))
            await session.run()
        except:
            traceback.print_exc()

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
        use_tts: bool = False,
    ):
        for component in service.required_components:
            setattr(self, component, service.components[component])

        self.workspace = (
            Path(tempfile.TemporaryDirectory().name)
            if not workspace
            else Path(workspace)
        )
        self.workspace.mkdir(parents=True, exist_ok=True)

        self.data = b""
        self.service = service
        self.session_id = session_id
        self.sample_rate = 16000

        self.vad_states = None
        self.voiced_segments: List[Dict] = []
        self.transcripts = []
        self.chat_messages: List[Dict] = []
        self.chatbot_controller = None
        self.text2speech_controller = None

        self.user_audio_input_stream = user_audio_input_stream
        self.agent_audio_output_stream = agent_audio_output_stream

        self.allow_vad_interrupt = allow_vad_interrupt
        self.use_tts = use_tts

        self.speech2text_min_bytes = int(SPEECH2TEXT_MIN_SECONDS * self.sample_rate * 2)
        self.speech2text_max_bytes = int(SPEECH2TEXT_MAX_SECONDS * self.sample_rate * 2)
        self.speech2text_min_interval_bytes = int(
            SPEECH2TEXT_MIN_INTERVAL_SECONDS * self.sample_rate * 2
        )
        self.speech2text_offset_bytes = 0
        self.speech2text_states = {"session_id": self.session_id}

        self.transcribe = self.transcribe_streaming if getattr(self.speech2text, "is_streaming", False) else self.transcribe_non_streaming

        self.voiced_bytes_to_interrupt = (
            int(VOICED_SECONDS_TO_INTERRUPT * self.sample_rate) * 2
        )
        self.unvoiced_bytes_to_eot = int(UNVOICED_SECONDS_TO_EOT * self.sample_rate) * 2

        self.event_bus = EventBus()
        self.agent_can_speak = False
        self.user_is_speaking = False
        self.event_bus.on(EventEnum.RECEIVE_ADUIO_CHUNK, self.on_receive_audio_chunk)
        self.event_bus.on(
            EventEnum.UPDATE_VOICED_SEGMENTS, self.on_update_voiced_segments
        )
        self.event_bus.on(EventEnum.INTERRUPT, self.on_interrupt)
        self.event_bus.on(EventEnum.END_OF_TURN, self.on_end_of_turn)

        self.last_vad_end = 0
        self.last_speech2text_end = 0

    async def run(self):
        asyncio.create_task(self.detect_speech())
        try:
            while True:
                chunk = await self.user_audio_input_stream.read(4096)
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
        if self.chatbot_controller:
            self.chatbot_controller.send("interrupt")
            self.chatbot_controller.close()
            self.chatbot_controller = None
        if self.text2speech_controller:
            # the text2speech_controller might already be closed
            try:
                self.text2speech_controller.send("interrupt")
                self.text2speech_controller.close()
            except OSError as e:
                if e.errno != errno.EPIPE:
                    raise e
                self.text2speech_controller = None

    async def on_end_of_turn(self, sender="user"):
        logger.debug(f"end of turn received from {sender}")
        if self.agent_can_speak:
            return

        await self.transcribe(end_of_stream=True)
        self.speech2text_states = {"session_id": self.session_id}
        transcript = self.transcript
        speech = self.speech
        if not transcript.strip():
            return

        self.clear_transcript()
        self.agent_audio_output_stream.reset()
        self.agent_can_speak = True
        logger.debug(f"agent start speaking")
        asyncio.create_task(self.agent_speak(speech, transcript))

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
                if (
                    self.voiced_segments
                    and self.voiced_segments[-1]["end"] >= result["start"] * 2
                ):
                    self.voiced_segments.pop()
                self.voiced_segments.append(
                    {
                        "start": result["start"] * 2
                        - int(VAD_LEFT_SILENCE_SECONDS * 16000 * 2),
                        "eos": False,
                    }
                )
            if "end" in result:
                self.user_is_speaking = False
                self.voiced_segments[-1]["end"] = max(self.voiced_segments[-1]["end"], result["end"] * 2)
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
            and not "eot" in self.voiced_segments[-1]
            and self.last_vad_end - self.voiced_segments[-1]["end"]
            > self.unvoiced_bytes_to_eot
        ):
            self.voiced_segments[-1]["eot"] = True
            await self.event_bus.emit(EventEnum.END_OF_TURN, sender="vad")
        elif (
            self.agent_can_speak
            and self.voiced_segments
            and self.last_vad_end == self.voiced_segments[-1]["end"]
            and self.voiced_segments[-1]["end"] - self.voiced_segments[-1]["start"]
            > self.voiced_bytes_to_interrupt
        ):
            await self.event_bus.emit(EventEnum.INTERRUPT, sender="vad")

    async def transcribe_non_streaming(self, end_of_stream=False):
        """Optimization in this functions applies to simulating online
        speech2text with offline speech2text model."""
        end, eos = self.voiced_segments[-1]["end"], self.voiced_segments[-1]["eos"]

        if end_of_stream:
            logger.debug("end of stream transcribe")
            assert eos, "received end_of_stream but eos not triggered"

        if end <= self.last_speech2text_end:
            return

        if not eos and not end_of_stream:
            if end - self.last_speech2text_end < self.speech2text_min_interval_bytes:
                return
            if len(self.speech) < self.speech2text_min_bytes:
                return

        if len(self.speech) > self.speech2text_max_bytes:
            if self.last_speech2text_end < self.voiced_segments[-1]["start"]:
                self.speech2text_offset_bytes = self.voiced_segments[-1]["start"]

        start, end = self.speech2text_offset_bytes, self.voiced_segments[-1]["end"]
        self.last_speech2text_end = max(end, self.last_speech2text_end)
        text, self.speech2text_states = await asyncio.to_thread(self.speech2text.process, samples=self.speech, states=self.speech2text_states, end_of_stream=end_of_stream)
        logger.debug(f"text from {start} to {end}: {text}")
        self.update_transcript(start, end, text)

    async def transcribe_streaming(self, end_of_stream=False):
        start, end = self.speech2text_offset_bytes, self.voiced_segments[-1]["end"]
        text, self.speech2text_states = await asyncio.to_thread(self.speech2text.process, samples=self.speech, states=self.speech2text_states, end_of_stream=end_of_stream)
        logger.debug(f"text from {start} to {end}: {text}")
        self.update_transcript(start, end + end_of_stream, text)
        logger.info(f"transcript: {self.transcript}")

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
                {
                    "start": last_voiced_segment["end"],
                    "end": last_voiced_segment["end"],
                    "eos": False,
                }
            )

    async def agent_speak(self, speech: bytes, transcript):
        self.chatbot_controller, chatbot_controller_child = Pipe()
        self.text2speech_controller, text2speech_controller_child = Pipe()

        md5_hash = hashlib.md5(speech).digest()
        audio_id = uuid.UUID(bytes=md5_hash).hex

        # valid diarization
        # do_diar = self.diarization.is_diarizaition(speech, transcript)
        diarization = self.diarization.process_func(self.session_id, audio_id, speech, transcript)
        # if not do_diar:
        #     diarization = []

        self.chat_messages = self.chatbot_slm.add_user_message(
            self.chat_messages, audio_bytes=speech, sample_rate=self.sample_rate, audio_id=audio_id
        )

        # chatbot => chatbot_response_stream => text2speech_source_stream => text2speech
        chatbot_response_stream = FIFOTextIOStream()
        asyncio.create_task(
            asyncio.to_thread(
                self.chatbot_slm.process,
                messages=self.chat_messages[-CHATBOT_MAX_MESSAGES:],
                text_fifo_path=chatbot_response_stream.fifo_path.as_posix(),
                control_pipe=chatbot_controller_child,
                diarization=diarization
            )
        )


        control_params = {
            "session_id": self.session_id,
            "transcript": transcript,
            "speech": speech,
            "emotion": "default",
            "speed": "default",
            "voice": "default",
            "text_frontend": True,
            "stream": True,
        }

        text2speech_source_stream = FIFOTextIOStream()
        asyncio.create_task(
            asyncio.to_thread(
                self.text2speech.process,
                text_fifo_path=text2speech_source_stream.fifo_path.as_posix(),
                audio_fifo_path=self.agent_audio_output_stream.fifo_path.as_posix(),
                control_pipe=text2speech_controller_child,
                control_params=control_params,
            )
        )

        chatbot_response = ""
        chat_messages = self.chat_messages

        try:
            await text2speech_source_stream.write("")  # make sure the fifo is alive
            while True:
                message_partial = await chatbot_response_stream.read(10)
                if not message_partial:
                    break
                chatbot_response += message_partial
                self.chat_messages = self.chatbot_slm.add_agent_message(
                    chat_messages, chatbot_response
                )
                await text2speech_source_stream.write(message_partial)
            text2speech_source_stream.close()
        except Exception as e:
            self.agent_can_speak = False
            if isinstance(e, OSError):
                if e.errno != errno.EPIPE:
                    raise e
            logger.error(e)
            traceback.print_exc()

        # chatbot has finished generating, text2speech might still be processing
        if self.chatbot_controller:
            self.chatbot_controller.close()
            self.chatbot_controller = None
