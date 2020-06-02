#!/usr/bin/env python
from utils import kaldi_io
import numpy as np
import argparse
from scipy.io.wavfile import write
import os

def main_():
	ap = argparse.ArgumentParser()
	ap.add_argument("--bits", type=int, default=10)
	ap.add_argument("--sample-freq", type=int, default=22050)
	ap.add_argument("ali_rspecifier", type=str)
	ap.add_argument("wav_dir", type=str)

	args = ap.parse_args()

	num_class = 2 ** args.bits
	scale = 2 ** (16 - 1) - 1

	for utt, ali in kaldi_io.read_vec_int_ark(args.ali_rspecifier):
		ali_f = ali.astype(np.float32)
		sample = scale * (2 * ali_f / (num_class - 1.) - 1.)

		wav_path = os.path.join(args.wav_dir, "{}.wav".format(utt))
		write(wav_path, args.sample_freq, sample.astype(np.int16))

if __name__ == '__main__':
	main_()
