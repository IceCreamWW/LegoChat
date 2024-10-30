import logging
import os
from pathlib import Path
from typing import Dict

import numpy as np
import onnxruntime
import torch
from legochat.components import Component, register_component
from silero_vad import load_silero_vad
from silero_vad.utils_vad import VADIterator


@register_component("silero_vad")
class SileroVADComponent(Component):
    def __init__(self):
        pass

    def setup(self):
        logging.info("Loading Silero VAD model")
        model = load_silero_vad(onnx=True)
        sample_rate = 16000
        self.vad_iterator = VADIterator(model, sampling_rate=sample_rate)
        self.window_size_samples = 512

    def process_func(
        self,
        samples: bytes,
        prev_states: dict = None,
        end_of_stream: bool = False,
    ):
        if prev_states:
            samples = prev_states["buffer_samples"] + samples
            self.states = prev_states
        else:
            self.vad_iterator.reset_states()
        samples = torch.frombuffer(samples, dtype=torch.int16).float() / 32768.0

        states = {}
        results = []
        for i in range(0, len(samples), self.window_size_samples):
            if not end_of_stream and i + self.window_size_samples > len(samples):
                states["buffer_samples"] = samples[i:]
                break

            chunk = samples[i : i + self.window_size_samples]
            if len(chunk) < self.window_size_samples:
                chunk = torch.cat([chunk, torch.zeros(self.window_size_samples - len(chunk))])
            speech_dict = self.vad_iterator(chunk)
            if speech_dict:
                results.append(speech_dict)

        states["vad"] = self.states
        return results, states

    @property
    def states(self):
        states = {
            "triggered": self.vad_iterator.triggered,
            "temp_end": self.vad_iterator.temp_end,
            "current_sample": self.vad_iterator.current_sample,
            "model": {
                "_state": self.vad_iterator.model._state,
                "_context": self.vad_iterator.model._context,
                "_last_sr": self.vad_iterator.model._last_sr,
                "_last_batch_size": self.vad_iterator.model._last_batch_size,
            },
        }
        return states

    @states.setter
    def states(self, states: Dict):
        self.vad_iterator.triggered = states["triggered"]
        self.vad_iterator.temp_end = states["temp_end"]
        self.vad_iterator.current_sample = states["current_sample"]
        self.vad_iterator.model._state = states["model"]["_state"]
        self.vad_iterator.model._context = states["model"]["_context"]
        self.vad_iterator.model._last_sr = states["model"]["_last_sr"]
        self.vad_iterator.model._last_batch_size = states["model"]["_last_batch_size"]
        return states


if __name__ == "__main__":
    import numpy as np
    import soundfile as sf

    audio_file = "examples/audio.wav"
    data, sr = sf.read(audio_file)
    data = (data * 32768.0).astype(np.int16).tobytes()
    component = SileroVADComponent()
    results, states = component.process(data, None, True)
    print(results)
    breakpoint()
