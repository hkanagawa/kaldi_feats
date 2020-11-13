#!/usr/bin/env python

import argparse
import numpy as np
from utils import kaldi_io

if __name__ == "__main__":
	ap = argparse.ArgumentParser()
	ap.add_argument("scale", type=float)
	ap.add_argument("feat_rspecifier", type=str)
	ap.add_argument("feat_wspecifier", type=str)
	args = ap.parse_args()

	with kaldi_io.open_or_fd(args.feat_wspecifier, "wb") as feats_writer:
		for utt, feat in kaldi_io.read_mat_ark(args.feat_rspecifier):
			feat2 = feat * args.scale
			kaldi_io.write_mat(feats_writer, feat2, key=utt)
