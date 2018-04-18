# ssanchors

A python package for generating anchor stimuli, suitable for the subjective
evaluation of source separation algorithms using MUSHRA-type protocols.

**Beta Release - Not Fully Tested**

## Installation

For the time being, you will need to install the package using

```
pip install -r requirements.txt
```

## Usage

### Python

```python
from untwist.data.audio import Wave
import ssanchors

if __name__ == '__main__':

    target_source = Wave.read('audio/vox.wav')

    other_sources = [
        Wave.read('audio/drums.wav'),
        Wave.read('audio/bass.wav'),
        Wave.read('audio/other.wav')
    ]

    anchors = ssanchors.utilities.ensure_audio_doesnt_clip(
        [
            ssanchors.distorted_target(target_source),
            ssanchors.interference(target_source, other_sources),
            ssanchors.overall_quality(target_source, other_sources),
            ssanchors.target_sound_quality(target_source),
        ]
    )

    names = ['distorted_target',
             'interference',
             'overall_quality',
             'target_sound_quality']

    for name, anchor in zip(names, anchors):
        anchor.write('audio/{}.wav'.format(name))
```

### Command line

You can generate the default anchors using the command line tool:

```
ssanachors --all --target audio/vox.wav  --others audio/bass.wav audio/drums.wav audio/other.wav
```

For information on generating specific anchors:
```
ssanchors --help
```

## References

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
