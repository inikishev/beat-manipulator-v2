import typing
from collections import OrderedDict, abc

import numpy as np

from ..audio import Audio
from ..utils import get_next_key, interpolate, op_with_overflow, op_on_longest, op_on_shortest
from .common import post_step
from ..effects.effect import apply_effect
if typing.TYPE_CHECKING:
    from ..beatswap_ import Beatswap



def _apply_operation_to_beats(beats: list[np.ndarray], index: float, source: np.ndarray, operation, length_mode: str):
    """Applies an operation to existing beats. Returns modified beats and overflow."""

    if index < 0: index = len(beats) + index
    if index < 0: return beats, None
    if index > len(beats): return beats, None

    cur_beat_index = int(index)
    cur_beat = beats[cur_beat_index]
    cur_beat_sample = interpolate([0, cur_beat.shape[1]], index % 1)

    if length_mode == 'overflow':
        while True:
            # put audio onto current beat starting from current sample
            cur_beat[:, cur_beat_sample:], source = op_with_overflow(operation, cur_beat[:, cur_beat_sample:], source) # type:ignore
            beats[cur_beat_index] = cur_beat
            #cur_beat[:, cur_beat_sample:cur_beat_sample+arr.shape[1]-cur_src_sample] = operation(cur_beat[:, cur_beat_sample:cur_beat_sample+arr.shape[1]-cur_src_sample], arr[:, cur_src_sample:])

            # move cursor one source forward
            if source is None: return beats, None
            cur_beat_sample += source.shape[1]

            # sample overflows over this beat, move to next beat
            if cur_beat_sample >= cur_beat.shape[1]:
                cur_beat_sample = 0
                cur_beat_index += 1
                if cur_beat_index >= len(beats): return beats, source
                cur_beat = beats[cur_beat_index]

    elif length_mode == 'shortest':
        remainder = op_on_shortest(operation, cur_beat[:, cur_beat_sample:], source)
        beats[cur_beat_index] = np.concatenate([cur_beat[:, :cur_beat_sample], remainder], axis=1)
        return beats, None

    elif length_mode == 'longest':
        remainder = op_on_longest(operation, cur_beat[:, cur_beat_sample:], source)
        beats[cur_beat_index] = np.concatenate([cur_beat[:, :cur_beat_sample], remainder], axis=1)
        return beats, None

    elif length_mode == 'existing':
        remainder, _ = op_with_overflow(operation, cur_beat[:, cur_beat_sample:], source)
        beats[cur_beat_index] = np.concatenate([cur_beat[:, :cur_beat_sample], remainder], axis=1)
        return beats, None

    elif length_mode == 'new':
        remainder, _ = op_with_overflow(operation, source, cur_beat[:, cur_beat_sample:], )
        beats[cur_beat_index] = np.concatenate([cur_beat[:, :cur_beat_sample], remainder], axis=1)
        return beats, None

    else: raise ValueError(f'Invalid {length_mode = }')



def operation_beat(beatswap: "Beatswap", pattern: OrderedDict[typing.Any, dict[str, typing.Any]], key):
    """Adds a beat."""
    op = pattern[key]

    # get the source audio and make sure it is Audio
    source_name = op.get('source', '__main_audio__')
    if source_name in beatswap._sources:
        source = beatswap._sources[source_name]
    else:
        source = source_name
    if not isinstance(source, Audio):
        source = beatswap._sources[source_name] = Audio(source)

    # source mode (beat / seconds / samples)
    source_mode = op.get('source mode', 'beats')

    # slice source to make the new beat
    if source_mode == 'beats':

        # make sure beats are detected
        if source.beats is None: source.detect_beats()

        # get start and end
        if 'start' not in op: raise ValueError(f"{op} doesn't have `start` key")
        start = op['start']
        if 'stop' not in op: stop = start + op.get('length', 1)
        else: stop = op['stop']

        # slice
        if start >= len(source)-1 or stop >= len(source)-1: return False
        beat = source[start:stop]

    # slice the audio
    elif source_mode in ('seconds', 'samples'):
        start = op.get('start', 0)
        stop = op.get('stop', None)
        if stop is None:
            length = op.get('length', None)
            if length is not None: stop = start + length
        if source_mode == 'seconds':
            start = int(start * source.sr)
            if stop is not None: stop = int(stop * source.sr)

        beat = source.audio[start:stop].copy()

    else:
        raise ValueError(f"Source mode is not valid: {source_mode}")

    if 'effects' in op:
        for eff in op['effects']:
            func = eff['function']
            args = eff.get('args', ())
            kwargs = eff.get('kwargs', {})
            beat = apply_effect(beat, source.sr, effect = func, args = args, kwargs=kwargs)

    # add overflow if appending
    mode = op.get('mode', 'append')
    if mode == 'append':
        if beatswap._overflow is not None:
            beat, beatswap._overflow = op_with_overflow(np.add, beat, beatswap._overflow)
        if beatswap._mul_overflow is not None:
            beat, beatswap._mul_overflow = op_with_overflow(np.multiply, beat, beatswap._mul_overflow)

    # add the beat
    mode = op.get('mode', 'append')

    if mode == 'append':
        beatswap.beats.append(beat)

    elif mode == 'prepend':
        beatswap.beats.insert(0, beat)

    elif mode == 'insert':
        if 'index' not in op: raise ValueError(f"{op} with mode = 'insert' doesn't have `index` key")
        beatswap.beats.insert(op['index'], beat)

    elif mode == 'add':
        beatswap.beats, overflow = _apply_operation_to_beats(
            beats = beatswap.beats,
            index = op.get('index', -1),
            source = beat,
            operation = np.add,
            length_mode = op.get('length mode', 'overflow')
        )
        if beatswap._overflow is None: beatswap._overflow = overflow
        elif overflow is not None: beatswap._overflow = op_on_longest(np.add, beatswap._overflow, overflow)

    elif mode == 'multiply':
        if 'index' not in op: raise ValueError(f"{op} with mode = 'multiply' doesn't have `index` key")
        beatswap.beats, mul_overflow = _apply_operation_to_beats(
            beats = beatswap.beats,
            index = op.get('index', -1),
            source = beat,
            operation = np.multiply,
            length_mode = op.get('length mode', 'overflow')
        )
        if beatswap._mul_overflow is None: beatswap._mul_overflow = mul_overflow
        elif mul_overflow is not None: beatswap._mul_overflow = op_on_longest(np.multiply, beatswap._mul_overflow, mul_overflow)
    else:
        raise ValueError(f"Invalid mode: {mode}")

    # increment start/stop/step
    increment = op.get('increment', 0)
    if 'start' in op: op['start'] =  op['start'] + increment
    if 'stop' in op: op['stop'] = op['stop'] + increment
    if 'length' in op: op['length'] = op['length'] + increment

    return post_step(beatswap, pattern, key)