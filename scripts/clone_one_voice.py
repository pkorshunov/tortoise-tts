"""
From the set of target wav files it generates the correct speaker latent vector, 
then generates an audio file from a given text.

Usage:
    {0} [-v...] [options] <reference-text>
    {0} -h | --help
    {0} --version

Arguments:
    <reference-text>                Text which the voice will speak.
Options:
    -h --help                       Show this screen.
    --version                       Show version.
    -v, --verbose                   Increases the output verbosity level
    -f, --target-files STR          Glob to the set of target files.
    -o, --out-file STR              Path where the generated WAV file will be saved.
"""

import sys
import os
import glob
import numpy as np

from docopt import docopt

import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F


from tortoise.api import TextToSpeech
from tortoise.utils.audio import load_audio


def main():
    """ """
    # Collect command line arguments
    args = docopt(__doc__.format(sys.argv[0]), version="0.0.1")
    target_files = args['--target-files']
    text = args['<reference-text>']
    out_file = args['--out-file']
    
    target_paths = glob.glob(target_files)
    
    # This will download all the models used by Tortoise from the HF hub.
    tts = TextToSpeech()
    # Pick a "preset mode" to determine quality. Options: {"ultra_fast", "fast" (default), "standard", "high_quality"}. See docs in api.py
    preset = "high_quality"
    
    voice_samples = []
    # import ipdb; ipdb.set_trace()
    for audiopath in target_paths:
        c = load_audio(audiopath, 22050)
        voice_samples.append(c)
        
    gen = tts.tts_with_preset(text, voice_samples=voice_samples, conditioning_latents=None, preset=preset)
    torchaudio.save(out_file, gen.squeeze(0).cpu(), 24000)
        
    
if __name__ == "__main__":
    main()

