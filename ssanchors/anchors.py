'''
Anchor signals for a MUSHRA-like listening tests for assessing source
separation techniques. Five different anchors are provided for assessing the
perception regarding interference, distortion, artefacts, and overall quality
and target sound quality. The first three were designed following [1], the
overall quality anchor following [2] and the target sound quality following
[3].

[1] Emiya, V., Vincent, E., Harlander, N., & Hohmann, V. (2011).
    Subjective and Objective Quality Assessment of Audio Source
    Separation. IEEE TASLP, 19(7), 2046–2057.
    http://doi.org/10.1109/TASL.2011.2109381

[2] Cano, E., Fitzgerald, D., & Brandenburg, K. (2016).
    Evaluation of Quality of Sound Source Separation Algorithms:
    Human Perception vs Quantitative Metrics. In EUSIPCO
    (pp. 1758–1762).
    http://doi.org/10.1109/EUSIPCO.2016.7760550

[3] D. Ward, H. Wierstorf, R. D. Mason, E. M. Grais, and M. D. Plumbley, “BSS
    EVAL or PEASS? Predicting the Perception of Singing-Voice Separation,” in
    2018 IEEE International Conference on Acoustics, Speech and Signal
    Processing (ICASSP), Calgary, Canada, 2018.
    http://epubs.surrey.ac.uk/845998/
'''
from __future__ import division
import numpy as np
from . import utilities
from untwist.data import audio
import untwist.utilities


def distorted_target(target,
                     distortion_factor=0.2,
                     lowpass_cutoff=3500,
                     num_points=2048,
                     window_type='hann',
                     sample_rate=None):
    """
    Generates a distored target anchor, created by lowpass filtering the target
    source using a 3.5 kHz cutoff frequency and by randomly setting
    ``distortion_factor`` of the spectral frames (time slices) to zero.
    Finally, the anchor is loudness matched to the loudness of the target.

    Default parameters based on [1].

    Note that this code can't reproduce the distortion from [1] given their
    parameter values, and thus requires some tweaking.

    Parameters
    ----------

    target : np.ndarray or Wave, shape=(num_samples, num_channels)
        The true target source.
    distortion_factor : float, optional
        Proportion of time frames to zero out (default 0.2).
    lowpass_cutoff : float, optional
        Cutoff frequency of the lowpass filter in Hz (default 3500)
    num_points : int, optional
        Number of points to use for the FFT (default 2048).
    window_type : str, optional
        Type of window to use for the FFT (default hann).
    sample_rate : int, optional
        Only needed if Wave objects not provided (default None).

    Returns
    -------

    distorted_target_anchor : Wave, shape=(num_samples, num_channels)
        Lowpass filtered and time-distorted target source.
    """

    # Setup
    if not isinstance(target, audio.Wave):
        target = audio.Wave(target, sample_rate)

    stft, istft = utilities.stft_istft(num_points, window_type)

    # Processing
    x_fft = stft.process(target)

    if distortion_factor is not None:

        num_frames_to_remove = int(x_fft.num_frames * distortion_factor)

        idx = np.random.choice(x_fft.num_frames,
                               num_frames_to_remove,
                               replace=False)

        x_fft[:, idx] = 0

    if lowpass_cutoff is not None:
        cutoff = untwist.utilities.conversion.nearest_bin(lowpass_cutoff,
                                                          num_points,
                                                          target.sample_rate)
        x_fft[cutoff:] = 0

    distorted_target_anchor = istft.process(x_fft)[:target.num_frames]
    distorted_target_anchor.loudness = target.loudness

    return distorted_target_anchor


def musical_noise(target,
                  distortion_factor=0.99,
                  lowpass_cutoff=None,
                  num_points=2048,
                  window_type='hann',
                  sample_rate=None):
    """
    Generates a musical noise artefacts signal by randomly zeroing 99% of the
    time-frequency bins. You can optionally apply a lowpass filter.

    Default parameters based on [1].

    Parameters
    ----------

    target : np.ndarray or Wave, shape=(num_samples, num_channels)
        The true target source.
    distortion_factor : float, optional
        Proportion of time-frequency bins to zero out (default 0.99).
    lowpass_cutoff : float, optional
        Cutoff frequency of the lowpass filter in Hz (default None).
    num_points : int, optional
        Number of points to use for the FFT (default 2048).
    window_type : str, optional
        Type of window to use for the FFT (default hann).
    sample_rate : int, optional
        Only needed if Wave objects not provided (default None).

    Returns
    -------

    artefacts : Wave, shape=(num_samples, num_channels)
        Musical noise.
    """

    # Setup
    if not isinstance(target, audio.Wave):
        target = audio.Wave(target, sample_rate)

    stft, istft = utilities.stft_istft(num_points, window_type)

    # Processing
    x_fft = stft.process(target)

    idx = np.random.choice(
        x_fft.size,
        size=int(x_fft.size * distortion_factor),
        replace=False)
    freq, time, channel = np.unravel_index(idx, x_fft.shape)

    x_fft[freq, time, channel] = 0

    if lowpass_cutoff is not None:
        cutoff = untwist.utilities.conversion.nearest_bin(lowpass_cutoff,
                                                          num_points,
                                                          target.sample_rate)
        x_fft[cutoff:] = 0

    artefacts = istft.process(x_fft)

    return artefacts[:target.num_frames]


def artefacts(target,
              distortion_factor=0.99,
              lowpass_cutoff=None,
              num_points=2048,
              window_type='hann',
              sample_rate=None):
    """
    Generates an artefacts anchor by summing the target source with musical
    noise; both equally loud. Musical noise is created by randomly
    zeroing a proportion of the time-frequency bins (see ``musical_noise``).

    Default parameters based on [1].

    Parameters
    ----------

    target : np.ndarray or Wave, shape=(num_samples, num_channels)
        The true target source.
    distortion_factor : float, optional
        Proportion of time-frequency bins to zero out (default 0.99).
    lowpass_cutoff : float, optional
        Cutoff frequency of the lowpass filter in Hz (default None).
    num_points : int, optional
        Number of points to use for the FFT (default 2048).
    window_type : str, optional
        Type of window to use for the FFT (default hann).
    sample_rate : int, optional
        Only needed if Wave objects not provided (default None).

    Returns
    -------

    artefacts_anchor : Wave, shape=(num_samples, num_channels)
        Target source plus musical noise.
    """

    artefacts = musical_noise(target,
                              distortion_factor,
                              lowpass_cutoff,
                              num_points,
                              window_type,
                              sample_rate)

    artefacts.loudness = target.loudness

    artefacts_anchor = artefacts + target
    artefacts_anchor.loudness = target.loudness

    return artefacts_anchor


def interference(target,
                 others,
                 relative_loudness=0,
                 sample_rate=None):
    """
    Generates an interference anchor by summing the target signal and the sum
    of all other sources (the accompaniment). The loudness of the accompaniment
    relative to the target can be defined.

    Default parameters based on [1].

    Parameters
    ----------

    target : np.ndarray or Wave, shape=(num_samples, num_channels)
        The true target source.
    others : List or single np.ndarray or Wave object
        Each object should have the shape=(num_samples, num_channels)
        If a single array is given, this should correspond to the
        accompaniment.
    relative_loudness : float, optional
        Loudness of the accompaniment relative to the target (default 0,
        meaning equally loud).
    sample_rate : int, optional
        Only needed if Wave objects not provided.

    Returns
    -------

    interference_anchor : Wave, shape=(num_samples, num_channels)
        Target source plus loudness adjusted accompaniment.
    """

    target, accompaniment = utilities.target_accompaniment(target,
                                                           others,
                                                           sample_rate)

    if relative_loudness is not None:

        accompaniment.loudness = target.loudness + relative_loudness

    interference_anchor = target + accompaniment
    interference_anchor.loudness = target.loudness

    return interference_anchor


def overall_quality(target,
                    others,
                    distortion_factor_target=None,
                    distortion_factor_noise=0.99,
                    lowpass_cutoff_target=3500,
                    lowpass_cutoff_noise=None,
                    relative_loudness=0,
                    num_points=2048,
                    window_type='hann',
                    sample_rate=None):
    """
    Generates an overall quality anchor, defined as the sum of the lowpass
    filtered target, an artefacts signal and an interferering signal; all
    equally loud.

    Default parameters based on [2].

    Parameters
    ----------

    target : np.ndarray or Wave, shape=(num_samples, num_channels)
        The true target source.
    others : List or single np.ndarray or Wave object
        Each object should have the shape=(num_samples, num_channels)
        If a single array is given, this should correspond to the
        accompaniment.
    distortion_factor_target : float, optional
        Proportion of time frames to zero out (default None).
    distortion_factor_noise : float, optional
        Proportion of time-frequency bins to zero out (default 0.99). This
        determines the amount and timbre of musical noise.
    lowpass_cutoff_target : float, optional
        Cutoff frequency of the lowpass filter applied to the target in Hz
        (default 3500).
    lowpass_cutoff_noise : float, optional
        Cutoff frequency of the lowpass filter applied to the musical noise in
        Hz (default None).
    relative_loudness : float
        Loudness of the accompaniment relative to the distorted target (default
        0, meaning equally loud).
    num_points : int, optional
        Number of points to use for the FFT (default 2048)
    window_type : str, optional
        Type of window to use for the FFT (default hann)
    sample_rate : int, optional
        Only needed if Wave objects not provided (default None).

    Returns
    -------

    overall_quality_anchor : Wave, shape=(num_samples, num_channels)
        Lowpass filtered target plus musical noise plus loudness adjusted
        accompaniment.
    """

    target, accompaniment = utilities.target_accompaniment(target,
                                                           others,
                                                           sample_rate)

    signals_to_sum = [

        distorted_target(target,
                         distortion_factor_target,
                         lowpass_cutoff_target,
                         num_points,
                         window_type,
                         sample_rate),

        musical_noise(target,
                      distortion_factor_noise,
                      lowpass_cutoff_noise,
                      num_points,
                      window_type,
                      sample_rate),
        accompaniment
    ]

    for signal in signals_to_sum:
        signal.loudness = -23

    signals_to_sum[2].loudness = -23 + relative_loudness

    overall_quality_anchor = sum(signals_to_sum)[:target.num_frames]
    overall_quality_anchor.loudness = target.loudness

    return overall_quality_anchor


def target_sound_quality(target,
                         distortion_factor_target=0.2,
                         distortion_factor_noise=0.99,
                         lowpass_cutoff_target=3500,
                         lowpass_cutoff_noise=3500,
                         num_points=2048,
                         window_type='hann',
                         sample_rate=None):
    """
    Generates a target sound quality anchor, defined as the sum of the
    distorted target and an artefacts signal, both equally loud.

    Default parameters based on [3].

    Parameters
    ----------

    target : np.ndarray or Wave, shape=(num_samples, num_channels)
        The true target source.
    distortion_factor_target : float, optional
        Proportion of time frames to zero out (default 0.2).
    distortion_factor_noise : float, optional
        Proportion of time-frequency bins to zero out (default 0.99). This
        determines the amount and timbre of musical noise.
    lowpass_cutoff_target : float, optional
        Cutoff frequency of the lowpass filter applied to the target in Hz
        (default 3500).
    lowpass_cutoff_noise : float, optional
        Cutoff frequency of the lowpass filter applied to the musical noise in
        Hz (default 3500).
    num_points : int, optional
        Number of points to use for the FFT (default 2048).
    window_type : str, optional
        Type of window to use for the FFT (default hann).
    sample_rate : int, optional
        Only needed if Wave objects not provided (default None).

    Returns
    -------

    target_sound_quality_anchor : Wave, shape=(num_samples, num_channels)
        Lowpass filtered and time-distorted target source plus musical_noise.
    """

    if not isinstance(target, audio.Wave):
        target = audio.Wave(target, sample_rate)

    signals_to_sum = [

        distorted_target(target,
                         distortion_factor_target,
                         lowpass_cutoff_target,
                         num_points,
                         window_type,
                         sample_rate),

        musical_noise(target,
                      distortion_factor_noise,
                      lowpass_cutoff_noise,
                      num_points,
                      window_type,
                      sample_rate),
    ]

    for signal in signals_to_sum:
        signal.loudness = -23

    target_sound_quality_anchor = sum(signals_to_sum)[:target.num_frames]
    target_sound_quality_anchor.loudness = target.loudness

    return target_sound_quality_anchor
