#!/usr/bin/env python

import argparse
from utils import kaldi_io
import numpy as np

def main_():
	ap = argparse.ArgumentParser(usage="")

	# positional args
	ap.add_argument("feature_rspecifier", help="input feat")
	ap.add_argument("feature_wspecifier", help="output feat")
	
	args = ap.parse_args()
	with kaldi_io.open_or_fd(args.feature_wspecifier, "wb") as feature_writer:
		for key, feats in kaldi_io.read_mat_ark(args.feature_rspecifier):
			if feats.shape[0] < 1:
				continue
			kaldi_io.write_mat(feature_writer, feats, key=key)

if __name__ == '__main__':
	main_()
