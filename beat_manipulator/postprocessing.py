import itertools

import numpy as np

def downbeat_consistency_fixedBPM(audio, downbeats):
    """Uses median downbeat length and finds the best downbeat to project median downbeats from.
    Puts 4 beats between each downbeat."""

    # step 1 - find the median downbeat length
    # not ideal because this biases smaller downbeats as there are more of them
    # we can also use downbeat length expectation as I pick a random position
    # but median is fine for now
    lengths = np.diff(downbeats)
    median_length = np.median(lengths)
    #print(f'{median_length = }')

    # step 2 - find the best middle point
    middlepoint_errors = []
    for d1,d2 in itertools.pairwise(downbeats):
        # project downbeats of median length from current downbeat
        projected_forward = np.arange(int(d1), audio.shape[1], median_length)
        projected_backward = np.arange(int(d1), 0, -median_length)
        projected = np.concatenate([projected_backward[::-1][:-1], projected_forward])

        # find the closest downbeat to each projected point
        diff_mat = np.abs(projected[:,None] - downbeats)
        closest = downbeats[np.argmin(diff_mat, axis = 1).astype(int)]

        # error between projected downbeats and their closest real downbeats
        middlepoint_errors.append(np.abs(projected - closest).sum())

    # lowest error downbeat wins
    best_point = downbeats[np.argmin(middlepoint_errors)]
    #print(f'{best_point = }')

    # project median length downbeats from best point
    projected_forward = np.arange(int(best_point), audio.shape[1], median_length)
    projected_backward = np.arange(int(best_point), 0, -median_length)
    proj_downbeats = np.concatenate([projected_backward[::-1][:-1], projected_forward])

    # now we put 4 beats between every projected downbeat
    proj_beats = []
    for d1,d2 in itertools.pairwise(proj_downbeats):
        downbeat_length = d2 - d1
        beat_length = downbeat_length / 4
        proj_beats.extend([d1, d1 + beat_length, d1 + beat_length * 2, d1 + beat_length * 3])

    return proj_beats, proj_downbeats

def beats_from_downbeats(audio, sr, downbeats):
    """puts 4 beats for every downbeat"""
    beats = []
    for d1,d2 in itertools.pairwise(downbeats):
        downbeat_length = d2 - d1
        beat_length = downbeat_length / 4
        beats.extend([d1, d1 + beat_length, d1 + beat_length * 2, d1 + beat_length * 3])

    return beats, downbeats