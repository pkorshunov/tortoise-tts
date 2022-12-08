"""
Given a file list from Bob dataset, and by using the original wav and text files, adapt to the voices 
and generate synthetic speech from the given text. 

Usage:
    {0} [-v...] [options] <probes-list>
    {0} -h | --help
    {0} --version

Arguments:
    <probes-list>                   A text file with the list of paths in Bob-db format to the original WAV files.
Options:
    -h --help                       Show this screen.
    --version                       Show version.
    -v, --verbose                   Increases the output verbosity level
    -i, --in-dir STR                Dir where the original WAV and text files are located.
    -o, --out-dir STR               Dir where to store generated files.
"""

#%% 
import sys
import os
import glob
import numpy as np
import pandas as pd


from docopt import docopt

from tortoise.api import TextToSpeech
from tortoise.utils.audio import load_audio

import torch
import torchaudio
# import torch.nn as nn
# import torch.nn.functional as F


def main():
    """ """
    # Collect command line arguments
    args = docopt(__doc__.format(sys.argv[0]), version="0.0.1")
    probes_file = args['<probes-list>']
    out_dir = args['--out-dir']
    in_dir = args['--in-dir']

    # This will download all the models used by Tortoise from the HF hub.
    tts = TextToSpeech()
    # Pick a "preset mode" to determine quality. Options: {"ultra_fast", "fast" (default), "standard", "high_quality"}. See docs in api.py
    preset = "fast"

    real_data = pd.read_csv(probes_file, na_values='nan')

    for cur_ref_id, cur_ref_id_group in real_data.groupby('REFERENCE_ID'):
        print(f"Processing speaker {cur_ref_id}...")
        list_of_files = cur_ref_id_group['PATH'].apply(lambda x: os.path.join(in_dir, x+'.wav')).to_list()
        list_of_txt = cur_ref_id_group['PATH'].apply(lambda x: os.path.join(in_dir, x+'.original.txt')).to_list()
        list_of_synth = cur_ref_id_group['ID'].apply(lambda x: os.path.join(out_dir, x+'_tortoise.wav')).to_list()
        
        # load WAV samples in the requred format
        voice_samples = list()
        for audiopath in list_of_files:
            c = load_audio(audiopath, 22050)
            voice_samples.append(c)
            
        # precompute latents for the current speaker
        auto_conditioning, diffusion_conditioning = tts.get_conditioning_latents(voice_samples, return_mels=False)
        
        # for each text, generate a synthetic WAV and save it
        for txt_file, out_file in zip(list_of_txt, list_of_synth):
            if os.path.exists(out_file):
                continue
            with open(txt_file, "r", encoding="utf-8") as file_text:
                text = file_text.readlines()[0]
            gen = tts.tts_with_preset(text, voice_samples=None, conditioning_latents=(auto_conditioning, diffusion_conditioning), preset=preset)
            os.makedirs(os.path.dirname(out_file), exist_ok=True)
            torchaudio.save(out_file, gen.squeeze(0).cpu(), 24000)
            
    
if __name__ == "__main__":
    main()

