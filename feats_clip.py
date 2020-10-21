#!/usr/bin/env python

import argparse
from utils import kaldi_io
import numpy as np

def main_():
	ap = argparse.ArgumentParser(usage="")

	# positional args
	ap.add_argument("--min", type=float, default=-1)
	ap.add_argument("--max", type=float, default=1)
	ap.add_argument("feature_rspecifier", help="input feat")
	ap.add_argument("feature_wspecifier", help="output feat")
	
	args = ap.parse_args()
	with kaldi_io.open_or_fd(args.feature_wspecifier, "wb") as feature_writer:
		for key, feats in kaldi_io.read_mat_ark(args.feature_rspecifier):
			new_feats = np.where(feats>=args.max, args.max, feats)
			new_feats = np.where(feats<=args.min, args.min, new_feats)
			kaldi_io.write_mat(feature_writer, new_feats, key=key)

if __name__ == '__main__':
	main_()
