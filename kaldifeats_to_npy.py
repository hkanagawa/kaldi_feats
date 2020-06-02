#!/usr/bin/env python
from utils import kaldi_io
import numpy as np
import argparse
import os

def main_():
	ap = argparse.ArgumentParser()
	ap.add_argument("feat_rspecifier", type=str)
	ap.add_argument("npy_outdir", type=str)

	args = ap.parse_args()
	npy_dir = args.npy_outdir
	for utt, feat in kaldi_io.read_mat_ark(args.feat_rspecifier):
		np.save(os.path.join(npy_dir, "{}.npy".format(utt)), feat.T)

if __name__ == '__main__':
	main_()
