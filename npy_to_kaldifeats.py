#!/usr/bin/env python
import kaldi_io
import numpy as np
import argparse
import sys

def main_():
	ap = argparse.ArgumentParser()
	ap.add_argument("npy_list", type=str)
	ap.add_argument("feat_wspecifier", type=str)

	args = ap.parse_args()

	npy_fp = open(args.npy_list, mode='rt') if args.npy_list != "-" else sys.stdin

	with kaldi_io.open_or_fd(args.feat_wspecifier, "wb") as feat_writer:
		for line in npy_fp:
			s = line.strip().split(" ")
			assert len(s) == 2
			utt, npy_file = s
			data = np.load(npy_file)
			kaldi_io.write_mat(feat_writer, data.T, key=utt)
		npy_fp.close()

if __name__ == '__main__':
	main_()
