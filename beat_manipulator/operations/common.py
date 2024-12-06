import random
import typing
from collections import OrderedDict, abc

import numpy as np

from ..audio import Audio
from ..utils import get_next_key, interpolate

if typing.TYPE_CHECKING:
    from ..beatswap_ import Beatswap


def _shuffle(beatswap: "Beatswap", ops, shuffle_key: str):
    shuffle_groups:dict[typing.Any, list] = {}

    # find all shuffle groups
    for i, key, op in ops:
        if shuffle_key in op:
            groups = op[shuffle_key]
            if not isinstance(groups, (list,tuple)): groups = [groups]

            for group in groups:
                if group not in shuffle_groups: shuffle_groups[group] = [i]
                shuffle_groups[group].append(i)

    if len(shuffle_groups) > 0:
        # shuffle all shuffle groups
        shuffled_shuffle_groups = {k: v.copy() for k, v in shuffle_groups.items()}
        for group in shuffled_shuffle_groups.values():
            random.shuffle(group)

        # assign shuffled groups to ops
        shuffled_ops = ops.copy()
        for group, indexes in shuffle_groups.items():
            shuffled_indexes = shuffled_shuffle_groups[group]
            for old_index, new_index in zip(indexes, shuffled_indexes):
                shuffled_ops[old_index] = ops[new_index]

        ops = shuffled_ops
        pattern = beatswap.pattern = OrderedDict({k:v for i,k,v in ops})

    else: pattern = beatswap.pattern
    return ops, pattern

def post_step(beatswap: "Beatswap", pattern: OrderedDict[typing.Any, dict[str, typing.Any]], key):
    op = pattern[key]
    ops = [(i, key, op) for i,(key,op) in enumerate(pattern.items())]

    # shuffle on first operation
    if beatswap._current_key == list(beatswap.pattern.keys())[0]:
        ops, pattern = _shuffle(beatswap, ops, 'shuffle group')

    # shuffle always
    ops, pattern = _shuffle(beatswap, ops, 'shuffle always group')

    # shuffle trigger
    # ...

    # advance to next operation
    if 'next' in op: beatswap._current_key = op['next']
    else: beatswap._current_key = get_next_key(beatswap.pattern, beatswap._current_key)

    return True