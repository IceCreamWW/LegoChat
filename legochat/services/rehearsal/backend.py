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

from dotenv import load_dotenv
from legochat.services.service import Service
from legochat.utils.event import EventBus, EventEnum
from legochat.utils.stream import AudioInputStream, AudioOutputStream, FIFOTextIOStream

logger = logging.getLogger("legochat")
env_file = Path(__file__).parent / ".env"
if env_file.exists():
    load_dotenv(env_file)

VOICED_SECONDS_TO_INTERRUPT = float(os.getenv("VOICED_SECONDS_TO_INTERRUPT", "0.3"))
UNVOICED_SECONDS_TO_EOT = float(os.getenv("UNVOICED_SECONDS_TO_EOT", "0.6"))
CHATBOT_MAX_MESSAGES = int(os.getenv("CHATBOT_MAX_MESSAGES", "20"))


class ChatSpeech2Speech(Service):
    def __init__(self, config):
        super().__init__(config)
        self.sessions = {}

    @property
    def required_components(self):
        return ["vad", "chatbot_slm", "text2speech"]

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
        model: str,
        workspace: Optional[Path] = None,
        allow_vad_interrupt: bool = True,
    ):
        for component in service.required_components:
            setattr(self, component, service.components[component])

        self.workspace = (
            Path(tempfile.TemporaryDirectory().name)
            if not workspace
            else Path(workspace)
        )
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.scene = None

        self.data = b""
        self.service = service
        self.session_id = session_id
        self.sample_rate = 16000

        self.vad_states = None
        self.voiced_segments: List[Dict] = []
        self.chatbot_controller = None
        self.chat_messages: List[Dict] = []
        self.text2speech_controller = None

        self.user_audio_input_stream = user_audio_input_stream
        self.agent_audio_output_stream = agent_audio_output_stream

        self.allow_vad_interrupt = allow_vad_interrupt

        self.speech2text_offset_bytes = 0
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
        pass

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

    def on_end_of_turn(self, sender="user"):
        logger.debug(f"end of turn received from {sender}")
        if self.agent_can_speak:
            return

        user_speech = self.speech
        self.clear_user_speech()
        self.agent_audio_output_stream.reset()
        self.agent_can_speak = True
        asyncio.create_task(self.agent_speak(user_speech))

    def clear_user_speech(self):
        if not self.voiced_segments:
            return
        last_voiced_segment = self.voiced_segments[-1]
        self.voiced_segments.clear()

        # handle manual EOT when VAD has not detect eos yet
        if not last_voiced_segment["eos"]:
            self.voiced_segments.append(
                {
                    "start": last_voiced_segment["end"],
                    "end": last_voiced_segment["end"],
                    "eos": False,
                }
            )

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
                    {"start": result["start"] * 2, "eos": False}
                )
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
            and self.last_vad_end - self.voiced_segments[-1]["end"]
            > self.unvoiced_bytes_to_eot
        ):
            await self.event_bus.emit(EventEnum.END_OF_TURN, sender="vad")
        elif (
            self.agent_can_speak
            and self.voiced_segments
            and self.last_vad_end == self.voiced_segments[-1]["end"]
            and self.voiced_segments[-1]["end"] - self.voiced_segments[-1]["start"]
            > self.voiced_bytes_to_interrupt
        ):
            await self.event_bus.emit(EventEnum.INTERRUPT, sender="vad")

    @property
    def speech(self):
        data = b""
        for segment in self.voiced_segments:
            if segment["start"] >= self.speech2text_offset_bytes:
                data += self.data[segment["start"] : segment["end"]]
        return data

    async def agent_speak(self, user_speech: bytes):
        self.chatbot_controller, chatbot_controller_child = Pipe()
        self.text2speech_controller, text2speech_controller_child = Pipe()

        self.chat_messages = self.chatbot_slm.add_user_message(
            self.chat_messages, audio_bytes=user_speech, sample_rate=self.sample_rate
        )

        if len(self.chat_messages) > CHATBOT_MAX_MESSAGES:
            self.chat_messages = self.chat_messages[-CHATBOT_MAX_MESSAGES:]

        # chatbot => chatbot_response_stream => text2speech_source_stream => text2speech
        ith_message = (
            len(
                [message for message in self.chat_messages if message["role"] == "user"]
            )
            - 1
        )
        chatbot_response_stream = FIFOTextIOStream()
        asyncio.create_task(
            asyncio.to_thread(
                self.chatbot_slm.process,
                messages=self.chat_messages,
                text_fifo_path=chatbot_response_stream.fifo_path.as_posix(),
                control_pipe=chatbot_controller_child,
                scene=self.scene,
                ith_message=ith_message,
            )
        )

        text2speech_source_stream = FIFOTextIOStream()
        asyncio.create_task(
            asyncio.to_thread(
                self.text2speech.process,
                text_fifo_path=text2speech_source_stream.fifo_path.as_posix(),
                audio_fifo_path=self.agent_audio_output_stream.fifo_path.as_posix(),
                control_pipe=text2speech_controller_child,
                ith_message=ith_message,
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

        # chatbot has finished generating, text2speech might still be processing
        if self.chatbot_controller:
            self.chatbot_controller.close()
            self.chatbot_controller = None