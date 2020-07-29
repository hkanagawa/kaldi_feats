#!/usr/bin/env python
import argparse
import pickle
import sys

def main_():
	ap = argparse.ArgumentParser()
	ap.add_argument("utt_rxfilename", type=str)
	ap.add_argument("datapkl", type=str)

	args = ap.parse_args()
	utt_fp = open(args.utt_rxfilename, mode='rt') if args.utt_rxfilename != "-" else sys.stdin

	uttids = []
	for line in utt_fp:
		s = line.strip().replace("\t", " ").split(" ")
		assert len(s) == 1
		uttids.append(s[0])

	pkl_fp = open(args.datapkl, mode='wb') if args.datapkl != "-" else sys.stdout
	pickle.dump(uttids, pkl_fp)

if __name__ == '__main__':
	main_()
