import typing
from collections import abc
from collections import OrderedDict

import numpy as np


from .operations.beat import operation_beat
from .audio import Audio

OPERATIONS = {"beat": operation_beat}

class Beatswap:
    def __init__(self, pattern: dict[typing.Any, dict[str, typing.Any]], sources: dict[typing.Any, typing.Any]):
        self.pattern = OrderedDict(pattern)
        self._current_key = list(self.pattern.keys())[0]

        self.beats:list[np.ndarray] = []
        self._overflow: np.ndarray | None = None
        self._mul_overflow: np.ndarray | None = None
        self._can_continue = True
        self._variables = {}

        self._sources: dict[typing.Any, typing.Any] = sources

    def step(self):
        op = self.pattern[self._current_key]

        # perform the operation
        operation_type = op.get('operation', 'beat')
        self._can_continue = OPERATIONS[operation_type](self, self.pattern, self._current_key)

    def run(self):
        while self._can_continue:
            self.step()

        return np.concatenate(self.beats, axis = 1)


def beatswap(song, pattern:str, increment: float, sr = None):
    """Temporary simple pattern parser for testing"""
    beats = pattern.replace(' ', '').split(',')
    ops = OrderedDict()
    for i, b in enumerate(beats):
        if ":" in b:
            start, stop = b.split(":")
            ops[i] = {"start": float(start), "stop": float(stop), "increment": increment}
        else:
            ops[i] = {"start": float(b), "increment": increment}

    bs = Beatswap(ops, {"__main_audio__": Audio(song, sr)})
    return bs.run()

def beatswap_dict(song, pattern:dict, sr = None):
    bs = Beatswap(pattern, {"__main_audio__": Audio(song, sr)})
    return bs.run()
