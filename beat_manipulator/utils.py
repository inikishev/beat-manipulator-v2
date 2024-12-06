import math
from collections import abc
import numpy as np, torch

def get_next_key(d: dict, key):
    """Returns key from `d` that is after `key`"""
    keys = list(d.keys())
    index = keys.index(key)
    if index == len(keys) - 1:
        return keys[0]
    else:
        return keys[index + 1]


def interpolate(seq: abc.Sequence[float] | np.ndarray, i: float):
    """Same as `seq[i]` but `i` can be float and it will interpolate the value"""
    if i < 0: raise ValueError(i)
    if i >= len(seq) - 1: return seq[-1]

    s = math.floor(i)
    if s == i: return seq[int(i)]

    e = math.ceil(i)
    return seq[s] + (seq[e] - seq[s]) * (i - s)


def tonumpy(x):
    if isinstance(x, torch.Tensor): return x.detach().cpu().numpy()
    elif isinstance(x, np.ndarray): return x
    else: return np.asarray(x)

def totensor(x):
    if isinstance(x, torch.Tensor): return x
    else: return torch.from_numpy(tonumpy(x))


def op_with_overflow(op, beat:np.ndarray, audio: np.ndarray):
    """Applies `op` which adds `audio` to `beat`. If `beat` is longer than audio, remainder is the overflow."""

    # audio is longer
    if audio.shape[1] > beat.shape[1]:
        beat = op(beat, audio[:, :beat.shape[1]])
        return beat, audio[:, beat.shape[1]:]

    # beat is longer
    beat[:, :audio.shape[1]] = op(beat[:, :audio.shape[1]], audio)
    return beat, None


def op_on_longest(op, x1: np.ndarray, x2: np.ndarray):
    """Applies a SYMMETRIC operation on x1 and x2 which can be different lengths, returns longest."""
    if x1.shape[1] >= x2.shape[1]:
        x1[:, :x2.shape[1]] = op(x1[:, :x2.shape[1]], x2)
        return x1
    else:
        x2[:, :x1.shape[1]] = op(x2[:, :x1.shape[1]], x1)
        return x2
    
def op_on_shortest(op, x1: np.ndarray, x2: np.ndarray):
    """Applies a SYMMETRIC operation on x1 and x2 which can be different lengths, returns shortest."""
    if x1.shape[1] >= x2.shape[1]:
        x1[:, :x2.shape[1]] = op(x1[:, :x2.shape[1]], x2)
        return x1[:, :x2.shape[1]]
    else:
        x2[:, :x1.shape[1]] = op(x2[:, :x1.shape[1]], x1)
        return x2[:, :x1.shape[1]]