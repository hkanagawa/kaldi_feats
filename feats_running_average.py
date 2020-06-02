#!/usr/bin/env python

import argparse
import numpy as np
from utils import kaldi_io


def main_():
	ap = argparse.ArgumentParser(usage="")

	# positional args
	ap.add_argument("--window", default=3, type=int)
	ap.add_argument("feature_rspecifier", help="input feat")
	ap.add_argument("feature_wspecifier", help="output feat")

	args = ap.parse_args()
	avg_mask = np.ones(args.window) / args.window
	with kaldi_io.open_or_fd(args.feature_wspecifier, "wb") as feature_writer:
		for key, feats in kaldi_io.read_mat_ark(args.feature_rspecifier):
			dim = feats.shape[1]
			feats_avg = np.zeros(feats.shape, dtype=np.float)
			for d in range(dim):
				feats_avg[:, d] = np.convolve(feats[:, d], avg_mask, 'same')
			kaldi_io.write_mat(feature_writer, feats_avg, key=key)

if __name__ == '__main__':
	main_()
