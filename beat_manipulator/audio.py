import math
from collections import abc

import numpy as np

from .beat_detection import detect_beat_this
from .io_ import audioread
from .utils import interpolate, tonumpy


class Audio:
    """Holds audio and beats, slicing uses beats."""
    def __init__(self, audio, sr = None):
        if isinstance(audio, str):
            audio, sr = audioread(audio)

        else:
            audio = tonumpy(audio)
            if sr is None: sr = 44100

        self.audio: np.ndarray = audio
        self.sr = sr
        self.beats: np.ndarray | None = None
        self._cur_beat = 0

    def detect_beats(self):
        self.beats = detect_beat_this(self.audio, self.sr)

    def __getitem__(self, s: int | float | slice | abc.Iterable[int | float | slice | abc.Iterable]) -> np.ndarray:
        if self.beats is None:
            raise ValueError('Trying to slice Audio object that has no beats. Maybe you wanted to slice the array (self.audio)?')

        if isinstance(s, slice):
            start_idx, stop_idx = s.start, s.stop
            if s.step is not None: raise NotImplementedError('Step is not supported when slicing Audio.')

        elif isinstance(s, (int, float)):
            start_idx = s
            stop_idx = s+1

        else:
            return np.concatenate([self[beat] for beat in s], axis = 1)

        if start_idx is not None: start = int(interpolate(self.beats, start_idx))
        else: start = None

        if stop_idx is not None: stop = int(interpolate(self.beats, stop_idx))
        else: stop = None

        return self.audio[:, start:stop].copy()

    def __len__(self):
        if self.beats is None: raise ValueError('This Audio object has no beats')
        return len(self.beats)