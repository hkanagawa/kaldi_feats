#!/usr/bin/env python

import argparse
from utils import kaldi_io
import numpy as np

def main_():
	ap = argparse.ArgumentParser(usage="")

	# positional args
	ap.add_argument("uv_rspecifier", help="input uv")
	ap.add_argument("feature_rspecifier", help="input feat")
	ap.add_argument("feature_wspecifier", help="output feat")
	
	args = ap.parse_args()
	thresh = 0.5
	with kaldi_io.open_or_fd(args.uv_rspecifier, "rb") as uv_reader:
		with kaldi_io.open_or_fd(args.feature_wspecifier, "wb") as feature_writer:
			for key, feats in kaldi_io.read_mat_ark(args.feature_rspecifier):
				assert (key == kaldi_io.read_key(uv_reader))
				# import pdb
				# pdb.set_trace()
				uv = kaldi_io.read_mat(uv_reader)
				idx = np.where(uv < thresh)
				new_feats = np.delete(feats, idx[0], axis=0)
				if new_feats.shape[0] < 1:
					continue
				kaldi_io.write_mat(feature_writer, new_feats, key=key)
				del uv, idx, new_feats

if __name__ == '__main__':
	main_()
