import numpy as np

def audioread(path) -> tuple[np.ndarray, int]:
    """Returns `(channels, samples)` audio and sr"""
    import pedalboard.io
    with pedalboard.io.AudioFile(path, 'r') as f: # pylint:disable=E1129 # type:ignore
        audio = f.read(f.frames)
        sr = f.samplerate
    return audio, sr