#!/usr/bin/env python
from utils import kaldi_io
import numpy as np
import argparse
import os
import pickle

def main_():
	ap = argparse.ArgumentParser()
	ap.add_argument("--save_pickle", type=str, default="")
	ap.add_argument("ali_rspecifier", type=str)
	ap.add_argument("npy_outdir", type=str)

	args = ap.parse_args()
	npy_dir = args.npy_outdir
	uttids = []
	for utt, ali in kaldi_io.read_ali_ark(args.ali_rspecifier):
		uttids.append(utt)
		np.save(os.path.join(npy_dir, "{}.npy".format(utt)), ali)

	if args.save_pickle != "":
		with open(args.save_pickle, 'wb') as f:
			pickle.dump(uttids, f)

if __name__ == '__main__':
	main_()
