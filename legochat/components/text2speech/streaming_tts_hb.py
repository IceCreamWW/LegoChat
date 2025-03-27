import base64
import json
import logging
import re

from uuid import uuid4
import requests
from legochat.components import Component, register_component
from openai import OpenAI
from json import JSONDecodeError

logger = logging.getLogger("legochat")


def extract_tts_text(text):
    punctuation = r"[，。！？,.!?]"
    for i in range(len(text), 10, -1):
        prefix = text[:i]
        if re.search(punctuation + r"$", prefix) and len(prefix) > 10:
            return prefix, text[i:]
    return "", text


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


text2speech_control_prompt = [
            {
  "role": "system",
  "content": """
    You are a specialized agent to understand the input of a user and output the control signal that we will give for a Text to speech model to control the speed, timbre and emotion of the generated speech. You also need to give a control signal when the user want to summarize or ask questions of a specific speaker.
    Your task is to analyze user input and output a JSON object with 3 fields: speed, timbre, emotion. Follow these rules:\n\n
    The parameter values of speed is default, reset, fast and slow. 
    The parameter values of timbre: default, self, reset, male, female, child, nezha, taiyi, aobing. nezha means 哪吒, taiyi means 太乙真人, aobing 敖丙. 
    The parameter of emotion is default, reset, happy, angry, sad, surprise.
    taiyi is nezha's master.aobing is the dragon king's son. aobing and nezha are both enemies and friends. 
    When only these names appear and the user doesn't make any timbre control requests, set the timbre to the default.
    The parameter value of whether calling the extra speech diarization model to do summarize is True and False. 
    If input contains explicit instructions or related instructions for any parameter, use those values. If a parameter is mentioned but value is invalid, use default. If no instructions found, output default values. If partial instructions found, only update specified parameters\n\n
    Please always return valid JSON with all 3 fields and use lowercase for all values. Please Maintain strict format: {\"speed\": \"...\"). 
    Only use "nezha", "taiyi", or "aobing" for the timbre value when the user explicitly asks to clone their timbre. For other mimicry requests, use the default timbre unless 哪吒，太乙真人 and 敖丙 are explicitly mentioned. 
  """
},
{
    "role": "user",
    "content": "use the prompt of Antonie",
},
{
    "role": "assistant",
    "content": "{\"speed\": \"default\", \"timbre\": \"default\", \"emotion\": \"default\", \"diarization\": False}",
},

{
    "role": "user",
    "content": "试试看用特朗普的音色跟我说话",
},
{
    "role": "assistant",
    "content": "{\"speed\": \"default\", \"timbre\": \"default\", \"emotion\": \"default\", \"diarization\": False}",
},

{
    "role": "user",
    "content": "上一轮中有几个人在说话，分别说了什么内容",
},
{
    "role": "assistant",
    "content": "{\"speed\": \"default\", \"timbre\": \"default\", \"emotion\": \"default\", \"diarization\": True}",
},

{
    "role": "user",
    "content": "请帮我总结之前的几段对话",
},
{
    "role": "assistant",
    "content": "{\"speed\": \"default\", \"timbre\": \"default\", \"emotion\": \"default\", \"diarization\": True}",
},

{
    "role": "user",
    "content": "请让一个男人给我飞快地讲一个故事",
},
{
    "role": "assistant",
    "content": "{\"speed\": \"fast\", \"timbre\": \"male\", \"emotion\": \"default\", \"diarization\": False}",
},

{
    "role": "user",
    "content": "你恢复正常声音吧",
},
{
    "role": "assistant",
    "content": "{\"speed\": \"reset\", \"timbre\": \"reset\", \"emotion\": \"reset\", \"diarization\": False}",
},


{
    "role": "user",
    "content": "请用原来的速度跟我说话",
},
{
    "role": "assistant",
    "content": "{\"speed\": \"reset\", \"timbre\": \"default\", \"emotion\": \"default\", \"diarization\": False}",
},

{
    "role": "user",
    "content": "请用正常的速度和情感跟我说话",
},
{
    "role": "assistant",
    "content": "{\"speed\": \"reset\", \"timbre\": \"default\", \"emotion\": \"reset\", \"diarization\": False}",
},


{
    "role": "user",
    "content": "你要扮演太白金星，跟我来聊天",
},
{
    "role": "assistant",
    "content": "{\"speed\": \"default\", \"timbre\": \"default\", \"emotion\": \"default\", \"diarization\": False}",
},

{
    "role": "user",
    "content": "我想跟太乙真人对话，你能扮演一下她嘛",
},
{
    "role": "assistant",
    "content": "{\"speed\": \"default\", \"timbre\": \"taiyi\", \"emotion\": \"default\", \"diarization\": False}",
},

{
    "role": "user",
    "content": "克隆我的声音",
},
{
    "role": "assistant",
    "content": "{\"speed\": \"default\", \"timbre\": \"self\", \"emotion\": \"default\", \"diarization\": False}",
},

{
    "role": "user",
    "content": "模仿一下我生气的声音吧,快一点念哦",
},
{
    "role": "assistant",
    "content": "{\"speed\": \"fast\", \"timbre\": \"self\", \"emotion\": \"angry\", \"diarization\": False}",
},

{
    "role": "user",
    "content": "用敖丙的声音给我播报一个天气预报",
},
{
    "role": "assistant",
    "content": "{\"speed\": \"default\", \"timbre\": \"aobing\", \"emotion\": \"default\", \"diarization\": False}",
},

{
    "role": "user",
    "content": "你知道哪吒是谁吗",
},
{
    "role": "assistant",
    "content": "{\"speed\": \"default\", \"timbre\": \"default\", \"emotion\": \"default\", \"diarization\": False}",
},


{
    "role": "user",
    "content": "给我用生气的语气念一首诗歌",
},
{
    "role": "assistant",
    "content": "{\"speed\": \"default\", \"timbre\": \"default\", \"emotion\": \"angry\", \"diarization\": False}",
},

{
    "role": "user",
    "content": "湖水是蓝色的",
},
{
    "role": "assistant",
    "content": "{\"speed\": \"default\", \"timbre\": \"default\", \"emotion\": \"default\", \"diarization\": False}",
}]

@register_component("text2speech", "streaming_tts_hb")
class StreamingTTSHBComponent(Component):
    def __init__(self, url, llm_url):
        self.url = url
        self.llm_url = llm_url
        self.sample_rate = 16000
        self.client = OpenAI(base_url=llm_url, api_key="token")

    def setup(self):
        logger.info("StreamingTTSComponent setup")

    def tts(self, text, control_params):
        control_params = control_params.copy()
        files = {"ref_audio": pcm_to_wav(control_params.pop("speech"))}
        control_params["stream"] = True
        control_params["text_frontend"] = True
        control_params["gen_text"] = text
        control_params["ref_text"] = control_params.pop("transcript")
        control_params["session_id"] = control_params.pop("session_id")
        control_params["dtype"] = "np.int16"


        data = {"params": json.dumps(control_params)}
        with requests.post(self.url, files=files, data=data, stream=True) as response:
            for chunk in response.iter_content(chunk_size=4096):
                if chunk:
                    yield chunk

    def process_func(self, text_fifo_path, audio_fifo_path, control_pipe=None, control_params=None):
        messages = text2speech_control_prompt + [{"role": "user", "content": control_params["transcript"]}]
        completion = self.client.chat.completions.create(
            model="Qwen/Qwen2.5-7B-Instruct",
            messages=messages,
        )
        try:
            llm_control_params = json.loads(completion.choices[0].message.content)
            llm_control_params["voice"] = llm_control_params.pop("timbre")
        except JSONDecodeError as e:
            llm_control_params = {'speed': 'default', 'emotion': 'default', 'voice': 'default'}

        logger.debug(llm_control_params)
        control_params.update(llm_control_params)
        control_params["response_id"] = str(uuid4())

        text = ""
        with open(text_fifo_path, "r", encoding="u8") as fifo_text, open(
            audio_fifo_path, "wb"
        ) as fifo_audio:
            while True:
                text_partial = fifo_text.read(5)
                if not text_partial:
                    break
                if control_pipe and control_pipe.poll():
                    signal = control_pipe.recv()
                    if signal == "interrupt":
                        logger.debug("Streaming TTS process interrupted")
                        break
                text += text_partial
                tts_text, text = extract_tts_text(text)
                if tts_text:
                    for chunk in self.tts(tts_text, control_params=control_params):
                        fifo_audio.write(chunk)
            if text:
                for chunk in self.tts(text, control_params=control_params):
                    fifo_audio.write(chunk)
        if control_pipe:
            control_pipe.close()
        logger.debug("Streaming TTS Process Finished")
        return 0


if __name__ == "__main__":
    pass
