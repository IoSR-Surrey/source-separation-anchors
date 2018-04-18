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
