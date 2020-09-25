#!/usr/bin/env python
import argparse
from utils import kaldi_io
import numpy as np

def main_():
	ap = argparse.ArgumentParser()
	ap.add_argument("vec_rspecifier", type=str)
	ap.add_argument("feat_wspecifier", type=str)

	args = ap.parse_args()

	with kaldi_io.open_or_fd(args.feat_wspecifier, mode="wb") as feat_writer:
		for utt, vec in kaldi_io.read_vec_flt_ark(args.vec_rspecifier):
			feat = vec.reshape((-1, 1))
			kaldi_io.write_mat(feat_writer, feat, key=utt)

if __name__ == '__main__':
	main_()
