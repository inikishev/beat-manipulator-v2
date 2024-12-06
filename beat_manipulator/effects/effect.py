"""Effects. Note that all of them accept `audio` and `sr` as first 2 args, even if `sr` is not used."""
import numpy as np

def volume(audio: np.ndarray, sr: int, factor: float):
    """Multiplies volume"""
    return audio*factor

def speed(audio: np.ndarray, sr: int, factor: float = 2, precision:int = 48):
    """Changes the speed of the audio (if factor is not integer, it will be slightly inexact but fast)."""
    if factor%1 != 0 and (1/factor)%1 != 0:
        import fractions
        frac = fractions.Fraction(factor).limit_denominator(precision)
        audio = np.repeat(audio, frac.denominator, axis=1)
        return audio[:,::frac.numerator]

    elif factor%1 == 0:
        return audio[:,::int(factor)]

    else:
        return np.repeat(audio, int(1/factor), axis=1)

def channel(audio: np.ndarray, sr: int, ch:int | None = None):
    """If c is None, swaps channels. Otherwise zeroes `ch` channel."""
    if ch is None:
        audio[0], audio[1] = audio[1], audio[0]
        return audio

    audio[ch] = 0
    return audio

def downsample(audio: np.ndarray, sr: int, factor:int = 10):
    """Downsample by a factor of `d`"""
    return np.repeat(audio[:,::factor], factor, axis=1)

def gradient(audio: np.ndarray, sr: int, number: int = 1):
    """Takes the gradient of the audio multiple times"""
    for _ in range(number):
        audio = np.gradient(audio, axis=1)
    return audio

def bitcrush(audio: np.ndarray, sr: int, precision:float = 4):
    """Discretizes the audio to `precision` steps"""
    return np.around(audio*precision) / precision

def reverse(audio: np.ndarray, sr: int, ):
    return audio[:,::-1]

def normalize(audio: np.ndarray, sr: int, ):
    return audio*(1/np.max(np.abs(audio)))

def clip(audio: np.ndarray, sr: int, ):
    return np.clip(audio, -1, 1)

def pitch(audio: np.ndarray, sr: int, semitones):
    import pedalboard
    pitch_shift = pedalboard.PitchShift(semitones)
    return pitch_shift.process(audio, sample_rate = sr)

def stretch(audio: np.ndarray, sr: int, factor):
    import pedalboard
    return pedalboard.time_stretch(audio, sr, factor, )

def compress(audio: np.ndarray, sr: int, vbr_quality:float = 8):
    """Compresses the audio using mp3 compressor to `vbr_quality` which must be from 0 to 10 (lower = better quality)."""
    import pedalboard
    compressor = pedalboard.MP3Compressor(vbr_quality)
    return compressor(audio, sr)

def reverb(
    audio: np.ndarray,
    sr: int,
    room_size: float = 0.5,
    damping: float = 0.5,
    wet_level: float = 0.33,
    dry_level: float = 0.4,
    width: float = 1,
    freeze_mode: float = 0,
):
    import pedalboard

    reverber = pedalboard.Reverb(
        room_size=room_size,
        damping=damping,
        wet_level=wet_level,
        dry_level=dry_level,
        width=width,
        freeze_mode=freeze_mode,
    )
    return reverber(audio, sr)


EFFECTS = {
    "vol": volume,
    "spd": speed,
    "chn": channel,
    "dwn": downsample,
    "grd": gradient,
    "bcr": bitcrush,
    "rev": reverse,
    "nrm": normalize,
    "clp": clip,
    "pit": pitch,
    "str": stretch,
    "cmp": compress,
    "rvb": reverb
}

def apply_effect(audio, sr, effect, args, kwargs):
    fn = EFFECTS[effect]
    return fn(audio, sr, *args, **kwargs)