#!/usr/bin/env python
import numpy as np
from argparse import ArgumentParser
from utils import kaldi_io

def main_():
	ap = ArgumentParser()
	ap.add_argument("feat_rspecifier", type=str)
	ap.add_argument("ali_wspecifier", type=str)

	args = ap.parse_args()

	with kaldi_io.open_or_fd(args.ali_wspecifier, mode="wb") as ali_writer:
		for utt, feat in kaldi_io.read_mat_ark(args.feat_rspecifier):
			T, dim = feat.shape
			assert (dim == 1)
			ali = feat.squeeze(1).astype(np.int)
			kaldi_io.write_vec_int(ali_writer, ali, key=utt)

if __name__ == '__main__':
	main_()
