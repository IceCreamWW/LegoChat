import base64
import json
import logging
import re
from typing import Optional

from uuid import UUID
import requests
from legochat.components import Component, register_component
from openai import OpenAI
import json_repair
import hashlib

logger = logging.getLogger("legochat")

def pcm_to_wav(pcm_bytes, sample_rate=16000, num_channels=1, bits_per_sample=16):
    import io
    import wave

    byte_io = io.BytesIO()
    with wave.open(byte_io, "wb") as wav_file:
        wav_file.setnchannels(num_channels)
        wav_file.setsampwidth(bits_per_sample // 8)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_bytes)
    return byte_io.getvalue()


diar_control_prompt = [
    {
        "role": "system",
        "content": 'You are a specialized agent responsible for determining whether to call the speech diarization model for conversation summarization or answering speaker-specific questions. Based on the user input, output a JSON object with a single field: "diarization". The value must be either true or false, using lowercase letters. Always return valid JSON, e.g., {"diarization": false}.',
    },
    {"role": "user", "content": "刚才的对话里，一共有多少个人说话，分别说了什么？"},
    {"role": "assistant", "content": '{"diarization": true}'},
    {"role": "user", "content": "请总结我们的对话内容。"},
    {"role": "assistant", "content": '{"diarization": true}'},
    {"role": "user", "content": "刚才的对话中，我说了什么？"},
    {"role": "assistant", "content": '{"diarization": true}'},
    {"role": "user", "content": "刚才的对话中，韩冰有什么想法？"},
    {"role": "assistant", "content": '{"diarization": true}'},
    {"role": "user", "content": "告诉我未来24小时的天气。"},
    {"role": "assistant", "content": '{"diarization": false}'},
]


@register_component("diarization", "offline")
class OfflineDiarization(Component):
    def __init__(self, url, llm_url, min_speaker_num=1, max_speaker_num=2, speaker_num = None):
        self.sample_rate = 16000
        self.url = url
        self.client = OpenAI(base_url=llm_url, api_key="token")
        self.min_speaker_num = min_speaker_num
        self.max_speaker_num = max_speaker_num
        self.speaker_num = speaker_num

    def setup(self):
        logger.info("diarization setup")

    def is_diarizaition(
        self, audio_bytes: bytes, transcript: Optional[str] = None
    ):
        if transcript:
            messages = diar_control_prompt + [{"role": "user", "content": transcript}]
        elif audio_bytes:
            messages = diar_control_prompt + [
                {
                    "role": "user",
                    "content": {
                        "type": "input_audio",
                        "input_audio": {"data": audio_bytes, "format": "wav"},
                    },
                }
            ]
        else:
            raise

        completion = self.client.chat.completions.create(
            model="gpt-4o-audio",
            messages=messages,
        )

        try:
            control_params = json_repair.loads(completion.choices[0].message.content)
            logger.debug(">>> control diarization", control_params)
            return control_params.get("diarization", False)
        except Exception as e:
            logger.error(f"Error parsing JSON: {e}")
            return False

    def process_func(
        self, session_id, audio_id, audio_bytes, transcript="",
    ) -> dict[str, int]:
        params = {
            "session_id": session_id,
            "sent_id": audio_id,
            "transcript": transcript,
            "min_spk": self.min_speaker_num,
            "max_spk": self.max_speaker_num,
            "num_spk": self.speaker_num,
            "suffix": "wav",
        }

        try:
            response = requests.post(
                self.url, files={"new_audio": pcm_to_wav(audio_bytes)}, data={"params": json.dumps(params)}
            )
            logger.debug(">>> get diarization response", response)
            return response.json()
        except Exception as e:
            logger.error(f"Error in diarization: {e}")
            return {}


if __name__ == "__main__":
    diar = OfflineDiarization(url="http://localhost:8000", llm_url="http://localhost:8001")

