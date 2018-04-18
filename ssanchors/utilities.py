from __future__ import division
import numpy as np
from untwist import data
from untwist import transforms


def target_accompaniment(target, others, sample_rate=None):
    """

    Given a target source and list of 'other' sources, this function returns
    the target and accompaniment as untwist.data.audio.Wave objects. The
    accompaniment is defined as the sum of the other sources.

    Parameters
    ----------

    target : np.ndarray or Wave, shape=(num_samples, num_channels)
        The true target source.
    others : List or single np.ndarray or Wave object
        Each object should have the shape=(num_samples, num_channels)
        If a single array is given, this should correspond to the
        accompaniment.
    sample_rate : int, optional
        Only needed if Wave objects not provided.

    Returns
    -------
    target : Wave, shape=(num_samples, num_channels)
    accompaniment : Wave, shape=(num_samples, num_channels)

    """

    if isinstance(others, list):

        if not isinstance(others[0], data.audio.Wave):
            others = [data.audio.Wave(_, sample_rate) for _ in others]

        accompaniment = sum(other for other in others)

    else:

        if not isinstance(others, data.audio.Wave):
            others = data.audio.Wave(others, sample_rate)

        accompaniment = others

    if not isinstance(target, data.audio.Wave):
        target = data.audio.Wave(target, sample_rate)

    return target, accompaniment


def stft_istft(num_points=2048, window='hann'):
    """

    Returns an STFT and an ISTFT Processor object, both configured with the
    same window and transform length. These objects are to be used as follows:

        >>> stft, istft = stft_istft()
        >>> x = untwist.data.audio.Wave.tone() # Or some Wave
        >>> y = stft.process(x)
        >>> x = istft.process(y)

    Parameters
    ----------

    num_points : int
        The number of points to use for the window and the fft transform.
    window : str
        The type of window to use.

    Returns
    -------
    stft : untwist.transforms.stft.STFT
        An STFT processor.
    itft : untwist.transforms.stft.ITFT
        An ISTFT processor.
    """

    stft = transforms.STFT(window, num_points, num_points // 2)
    istft = transforms.ISTFT(window, num_points, num_points // 2)

    return stft, istft


def ensure_audio_doesnt_clip(list_of_arrays):
    """

    Takes a list of arrays and scales them by the same factor such that
    none clip.

    Parameters
    ----------

    list_of_arrays : list
        A list of array_like objects

    Returns
    -------

    new_list_of_arrays : list
        A list of scaled array_like objects.
    """

    max_peak = 1
    for audio in list_of_arrays:
        audio_peak = np.max(np.abs(audio))
        if audio_peak > max_peak:
            max_peak = audio_peak

    if max_peak >= 1:

        print('Warning: Audio has been attenuated to prevent clipping')

        gain = 0.999 / max_peak
        new_list_of_arrays = []
        for audio in list_of_arrays:
            new_list_of_arrays.append(audio * gain)
    else:

        new_list_of_arrays = list_of_arrays

    return new_list_of_arrays
