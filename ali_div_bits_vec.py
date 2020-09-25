#!/usr/bin/env python
import argparse
from utils import kaldi_io
import numpy as np

def main_():
	ap = argparse.ArgumentParser()
	ap.add_argument("--bits", type=int, default=16)
	ap.add_argument("ali_rspecifier", type=str)
	ap.add_argument("vec_wspecifier", type=str)

	args = ap.parse_args()
	den = 2 ** (args.bits - 1)

	with kaldi_io.open_or_fd(args.vec_wspecifier, mode="wb") as vec_writer:
		for utt, ali in kaldi_io.read_ali_ark(args.ali_rspecifier):
			ali2 = ali.astype(dtype=np.float32) / den
			kaldi_io.write_vec_flt(vec_writer, ali2, key=utt)

if __name__ == '__main__':
	main_()
