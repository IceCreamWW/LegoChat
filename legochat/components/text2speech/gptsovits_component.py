"""This file hacks to work with the GPT-SoVITS official repo, DO NOT try to
optimize it."""

import logging
import os
import sys
import time

sys.path.append("/mnt/disk2/home/vv/workspace/text_to_speech/GPT-SoVITS/develop/240811/GPT-SoVITS/GPT_SoVITS")

os.environ["G2PW_MODEL_DIR"] = (
    "/mnt/disk2/home/vv/workspace/text_to_speech/GPT-SoVITS/develop/240811/GPT-SoVITS/GPT_SoVITS/text/G2PWModel"
)

os.environ["G2PW_MODEL_SOURCE"] = (
    "/mnt/disk2/home/vv/workspace/text_to_speech/GPT-SoVITS/develop/240811/GPT-SoVITS/GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
)


import re
from pathlib import Path

import LangSegment
import numpy as np
import soundfile as sf
import torch
from AR.models.t2s_model import Text2SemanticDecoder
from legochat.components import Component, register_component
from module.models import SynthesizerTrn
from text import chinese, cleaned_text_to_sequence
from text.cleaner import clean_text
from transformers import AutoModelForMaskedLM, AutoTokenizer

logger = logging.getLogger("legochat")


def extract_tts_text(text):
    punctuation = r"[，。！？,.!?]"
    for i in range(len(text), 5, -1):
        prefix = text[:i]
        if re.search(punctuation + r"$", prefix) and len(prefix) > 5:
            return prefix, text[i:]
    return "", text


class InferenceUtils:
    @classmethod
    def setup(cls):
        if hasattr(cls, "device"):  # already setup
            return
        cls.version = os.environ.get("version", "v2")
        cnhubert_base_path = os.environ.get(
            "cnhubert_base_path",
            "/mnt/disk2/home/vv/workspace/text_to_speech/GPT-SoVITS/develop/240811/GPT-SoVITS/GPT_SoVITS/pretrained_models/chinese-hubert-base",
        )
        bert_path = os.environ.get(
            "bert_path",
            "/mnt/disk2/home/vv/workspace/text_to_speech/GPT-SoVITS/develop/240811/GPT-SoVITS/GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large",
        )
        cls.is_half = eval(os.environ.get("is_half", "True")) and torch.cuda.is_available()
        cls.language = os.environ.get("language", "Auto")
        if torch.cuda.is_available():
            cls.device = "cuda"
        else:
            cls.device = "cpu"

        cls.tokenizer = AutoTokenizer.from_pretrained(bert_path)
        cls.bert_model = AutoModelForMaskedLM.from_pretrained(bert_path)
        if cls.is_half == True:
            cls.bert_model = cls.bert_model.half().to(cls.device)
        else:
            cls.bert_model = cls.bert_model.to(cls.device)
        cls.dtype = torch.float16 if cls.is_half == True else torch.float32

    @classmethod
    def get_bert_feature(cls, text, word2ph):
        with torch.no_grad():
            inputs = cls.tokenizer(text, return_tensors="pt")
            for i in inputs:
                inputs[i] = inputs[i].to(cls.device)
            res = cls.bert_model(**inputs, output_hidden_states=True)
            res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]
        assert len(word2ph) == len(text)
        phone_level_feature = []
        for i in range(len(word2ph)):
            repeat_feature = res[i].repeat(word2ph[i], 1)
            phone_level_feature.append(repeat_feature)
        phone_level_feature = torch.cat(phone_level_feature, dim=0)
        return phone_level_feature.T

    @classmethod
    def clean_text_inf(cls, text, language, version):
        phones, word2ph, norm_text = clean_text(text, language, version)
        phones = cleaned_text_to_sequence(phones, version)
        return phones, word2ph, norm_text

    @classmethod
    def get_bert_inf(cls, phones, word2ph, norm_text, language):
        language = language.replace("all_", "")
        if language == "zh":
            bert = cls.get_bert_feature(norm_text, word2ph).to(cls.device)
        else:
            bert = torch.zeros((1024, len(phones)), dtype=cls.dtype).to(cls.device)

        return bert

    @classmethod
    def get_phones_and_bert(cls, text, language, version, final=False):
        if language in {"en", "all_zh", "all_ja", "all_ko", "all_yue"}:
            language = language.replace("all_", "")
            if language == "en":
                LangSegment.setfilters(["en"])
                formattext = " ".join(tmp["text"] for tmp in LangSegment.getTexts(text))
            else:
                # 因无法区别中日韩文汉字,以用户输入为准
                formattext = text
            while "  " in formattext:
                formattext = formattext.replace("  ", " ")
            if language == "zh":
                if re.search(r"[A-Za-z]", formattext):
                    formattext = re.sub(r"[a-z]", lambda x: x.group(0).upper(), formattext)
                    formattext = chinese.mix_text_normalize(formattext)
                    return cls.get_phones_and_bert(formattext, "zh", version)
                else:
                    phones, word2ph, norm_text = cls.clean_text_inf(formattext, language, version)
                    bert = cls.get_bert_feature(norm_text, word2ph).to(cls.device)
            elif language == "yue" and re.search(r"[A-Za-z]", formattext):
                formattext = re.sub(r"[a-z]", lambda x: x.group(0).upper(), formattext)
                formattext = chinese.mix_text_normalize(formattext)
                return cls.get_phones_and_bert(formattext, "yue", version)
            else:
                phones, word2ph, norm_text = cls.clean_text_inf(formattext, language, version)
                bert = torch.zeros(
                    (1024, len(phones)),
                    dtype=torch.float16 if cls.is_half == True else torch.float32,
                ).to(cls.device)
        elif language in {"zh", "ja", "ko", "yue", "auto", "auto_yue"}:
            textlist = []
            langlist = []
            LangSegment.setfilters(["zh", "ja", "en", "ko"])
            if language == "auto":
                for tmp in LangSegment.getTexts(text):
                    langlist.append(tmp["lang"])
                    textlist.append(tmp["text"])
            elif language == "auto_yue":
                for tmp in LangSegment.getTexts(text):
                    if tmp["lang"] == "zh":
                        tmp["lang"] = "yue"
                    langlist.append(tmp["lang"])
                    textlist.append(tmp["text"])
            else:
                for tmp in LangSegment.getTexts(text):
                    if tmp["lang"] == "en":
                        langlist.append(tmp["lang"])
                    else:
                        # 因无法区别中日韩文汉字,以用户输入为准
                        langlist.append(language)
                    textlist.append(tmp["text"])
            phones_list = []
            bert_list = []
            norm_text_list = []
            for i in range(len(textlist)):
                lang = langlist[i]
                phones, word2ph, norm_text = cls.clean_text_inf(textlist[i], lang, version)
                bert = cls.get_bert_inf(phones, word2ph, norm_text, lang)
                phones_list.append(phones)
                norm_text_list.append(norm_text)
                bert_list.append(bert)
            bert = torch.cat(bert_list, dim=1)
            phones = sum(phones_list, [])
            norm_text = "".join(norm_text_list)

        if not final and len(phones) < 6:
            return cls.get_phones_and_bert("." + text, language, version, final=True)

        return phones, bert.to(cls.dtype), norm_text


@register_component("gptsovits")
class GPTSoVITSComponent(Component):
    def __init__(self, speaker: str = "om"):
        self.ckpt_path = Path(
            f"/mnt/disk2/home/vv/workspace/text_to_speech/GPT-SoVITS/develop/240811/workspace/{speaker}/model.ckpt"
        )

    def setup(self):
        InferenceUtils.setup()
        self.device = InferenceUtils.device
        self.dtype = InferenceUtils.dtype
        stats = torch.load(self.ckpt_path, map_location="cpu")
        self._load_gpt(stats["gpt"])
        self._load_sovits(stats["sovits"])
        self._load_ref(stats["ref"])
        logger.info(f"GPT-SoVITS model loaded from {self.ckpt_path}")

    def _load_ref(self, stats):
        self.ref_spec = stats["spec"].to(self.dtype).to(self.device)
        self.ref_phones = stats["phones"]
        self.ref_bert = stats["bert"].to(self.dtype).to(self.device)
        self.ref_semantic = stats["semantic"].to(self.device)

    def _load_gpt(self, stats):
        self.t2s_model = Text2SemanticDecoder(stats["config"])
        self.t2s_model.load_state_dict(stats["weight"])
        self.t2s_model = self.t2s_model.to(self.device)
        if self.dtype == torch.float16:
            self.t2s_model = self.t2s_model.half()
        self.t2s_model.eval()
        self.early_stop_num = stats["config"]["early_stop_num"]

    def _load_sovits(self, stats):
        self.vq_model = SynthesizerTrn(**stats["config"]["model"])
        self.vq_model = self.vq_model.to(self.device)
        if self.dtype == torch.float16:
            self.vq_model = self.vq_model.half()
        self.vq_model.eval()
        self.vq_model.load_state_dict(stats["weight"], strict=False)
        self.sampling_rate = stats["config"]["data"]["sampling_rate"]

    def infer_segment(self, text, speed=1, top_k=20, top_p=0.6, temperature=0.6):
        if len(text.strip()) == 0:
            raise ValueError("text is empty")
        phones, bert, norm_text = InferenceUtils.get_phones_and_bert(text, language="auto", version="v2")
        bert = torch.cat([self.ref_bert, bert], 1)
        all_phoneme_ids = torch.LongTensor(self.ref_phones + phones).to(self.device).unsqueeze(0)
        bert = bert.to(self.device).unsqueeze(0)
        all_phoneme_len = torch.tensor([all_phoneme_ids.shape[-1]]).to(self.device)
        with torch.no_grad():
            pred_semantic, idx = self.t2s_model.infer_panel(
                all_phoneme_ids,
                all_phoneme_len,
                self.ref_semantic,
                bert,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                early_stop_num=self.early_stop_num,
            )
        pred_semantic = pred_semantic[:, -idx:].unsqueeze(0)
        audio = self.vq_model.decode(
            pred_semantic, torch.LongTensor(phones).to(self.device).unsqueeze(0), [self.ref_spec], speed=speed
        )
        audio = audio.cpu().numpy()[0, 0]
        if np.abs(audio).max() > 1:
            audio = audio / np.abs(audio).max() * 0.9
        return (audio, self.sampling_rate), norm_text

    def infer(self, text, speed=1, top_k=20, top_p=0.6, temperature=0.6):
        audio_segments = []
        silence = torch.zeros(int(self.sampling_rate * 0.3)).unsqueeze(0)
        norm_text = ""
        for segment in tts_split_text(text):
            segment = tts_fix_segment(segment)
            if len(segment.strip()) == 0:
                continue
            (audio_segment, sr), norm_segment = self.infer_segment(segment, speed, top_k, top_p, temperature)
            audio_segments.append(audio_segment)
            audio_segments.append(silence.numpy().squeeze(0))
            norm_text += norm_segment
        return (np.concatenate(audio_segments, 0), self.sampling_rate), norm_text

    @property
    def sample_rate(self):
        return 32000

    def process_func(self, text_fifo_path, audio_fifo_path, control_pipe=None):
        text = ""
        with open(text_fifo_path, "r", encoding="u8") as fifo_text, open(audio_fifo_path, "wb") as fifo_audio:
            while True:
                text_partial = fifo_text.read(5)
                if not text_partial:
                    break
                if control_pipe and control_pipe.poll():
                    signal = control_pipe.recv()
                    if signal == "interrupt":
                        logger.debug("GPTSoVITS process interrupted")
                        break
                text += text_partial
                tts_text, text = extract_tts_text(text)
                tts_text = tts_fix_segment(tts_text)
                if tts_text:
                    start = time.time()
                    (audio, sr), _ = self.infer(tts_text)
                    wav_bytes = (audio * 32768.0).astype(np.int16).tobytes()
                    end = time.time()
                    output_seconds = len(wav_bytes) / 2 / self.sample_rate
                    infernece_seconds = end - start
                    logger.debug(
                        f"GPTSoVITS [{tts_text}][{output_seconds:.3f}s]; cost {infernece_seconds:.3f}s, rtf: {infernece_seconds / output_seconds:.3f}"
                    )
                    fifo_audio.write(wav_bytes)
            if text:
                (audio, sr), _ = self.infer(text)
                wav_bytes = (audio * 32768.0).astype(np.int16).tobytes()
                fifo_audio.write(wav_bytes)

        if control_pipe:
            control_pipe.close()
        logger.debug("GPTSoVITS process finished")
        return 0


SENTENCE_SPLITS = ",.;?!、，。？！;：…)(（）“”《》'\"\t\n-"


def tts_fix_segment(segment):
    segment = re.sub(r"([" + SENTENCE_SPLITS + r"])+", r"\1", segment)
    if len(segment) == 0 or all(c in SENTENCE_SPLITS for c in segment):
        return ""
    if segment[-1] not in SENTENCE_SPLITS:
        segment += "."
    if len(segment) < 3 and segment[0] not in SENTENCE_SPLITS:
        segment = "." + segment

    return segment


def tts_split_text(text, min_segment_length=5, force_split_char="|"):
    segments = []
    segment = ""
    for c in text:
        do_split = False
        segment += c
        if c == force_split_char:
            segment = segment[:-1]
            do_split = True
        elif c in SENTENCE_SPLITS and len(segment) > min_segment_length:
            do_split = True

        if do_split:
            segments.append(segment)
            segment = ""

    if len(segment) > 0:
        segments.append(segment)

    if len(segments) >= 2 and all(c in SENTENCE_SPLITS for c in segments[-1]):
        segments[-2] += segments[-1]
        segments = segments[:-1]

    return segments


if __name__ == "__main__":
    text = "这个学期我们不开组会了，小组内啊，大家自己组织线上讨论。"
    synthesizer = GPTSoVITSComponent("om")
    synthesizer.setup()
    # wav_bytes = synthesizer.process_func(text)
    # sf.write("test.wav", audio, sr)
