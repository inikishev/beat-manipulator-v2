from .postprocessing import downbeat_consistency_fixedBPM, beats_from_downbeats
from .utils import totensor
import numpy as np

def detect_beat_this(audio, sr, dbn=True):
    """Detects beats with `beat_this`, returns list of beat positions in samples."""
    from beat_this.inference import Audio2Beats

    audio2beats = Audio2Beats(checkpoint_path="final0", device="cuda", dbn=dbn)
    beats, downbeats = audio2beats(totensor(audio.T), sr)
    beats = np.array(beats)*sr
    downbeats = np.array(downbeats)*sr
    #beats, downbeats = downbeat_consistency_fixedBPM(audio, np.array(downbeats)*sr)
    return beats