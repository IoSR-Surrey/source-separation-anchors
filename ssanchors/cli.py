from __future__ import division
import os
from collections import OrderedDict
import argparse
from untwist.data.audio import Wave
from .utilities import ensure_audio_doesnt_clip
from . import (distorted_target,
               artefacts,
               interference,
               target_sound_quality,
               overall_quality)


def ssanchors(argv=None):

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--target',
        type=str,
        help='A audio file representing the target source')

    parser.add_argument(
        '--others',
        nargs='+',
        help='A single or list of audio files representing the other sources')

    parser.add_argument(
        '--distorted_target',
        action='store_true',
        help='Generate the distorted target anchor')

    parser.add_argument(
        '--artefacts',
        action='store_true',
        help='Generate the artefacts anchor')

    parser.add_argument(
        '--interference',
        action='store_true',
        help='Generate the interference anchor')

    parser.add_argument(
        '--overall_quality',
        action='store_true',
        help='Generate the overall quality anchor')

    parser.add_argument(
        '--target_sound_quality',
        action='store_true',
        help='Generate the target sound quality anchor')

    parser.add_argument(
        '--all',
        action='store_true',
        help='Generates all anchors')

    args = parser.parse_args()

    '''
    Load audio and generate anchors
    '''

    target = Wave.read(args.target)

    if args.others is not None:
        others = [Wave.read(_) for _ in args.others]
    else:
        others = False

    anchors = OrderedDict()
    if args.distorted_target or args.all:
        anchors['distorted_target'] = distorted_target(target)

    if args.artefacts or args.all:
        anchors['artefacts'] = artefacts(target)

    if args.target_sound_quality or args.all:
        anchors['target_sound_quality'] = target_sound_quality(target)

    if args.interference or args.all:
        if others:
            anchors['interference'] = interference(target, others)
        else:
            print(
                'Cannot create interference anchor as '
                'no other sources provided'
            )

    if args.overall_quality or args.all:
        if others:
            anchors['overall_quality'] = overall_quality(target, others)
        else:
            print(
                'Cannot create overall quality anchor as '
                'no other sources provided'
            )

    keys, anchors = anchors.keys(), anchors.values()
    anchors = ensure_audio_doesnt_clip(anchors)

    filename, file_extension = os.path.splitext(args.target)

    for key, anchor in zip(keys, anchors):

        anchor.write(
                '{}_{}_anchor{}'
                .format(filename, key, file_extension)
            )


if __name__ == '__main__':

    ssanchors()
