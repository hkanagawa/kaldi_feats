#!/usr/bin/env python
from utils import kaldi_io
import numpy as np
import argparse
import os
import pickle

def main_():
	ap = argparse.ArgumentParser()
	ap.add_argument("--dim", default=1, type=int)
	ap.add_argument("feat_rspecifier", type=str)
	ap.add_argument("feat_wspecifier", type=str)

	args = ap.parse_args()
	with kaldi_io.open_or_fd(args.feat_wspecifier, "wb") as feature_writer:
		for utt, feat in kaldi_io.read_mat_ark(args.feat_rspecifier):
			length = feat.shape[0]
			new_feats = np.zeros((length, args.dim))
			
			kaldi_io.write_mat(feature_writer, new_feats, key=utt)

if __name__ == '__main__':
	main_()
