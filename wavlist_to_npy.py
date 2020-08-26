#!/usr/bin/env python
import wave
import numpy as np
import argparse
import sys

def main_():
	ap = argparse.ArgumentParser()
	ap.add_argument("wav_list", type=str)
	ap.add_argument("wav_outdir", type=str)

	args = ap.parse_args()

	wavlist_fp = open(args.wav_list, mode='rt') if args.wav_list != "-" else sys.stdin

	for line in wavlist_fp:
		s = line.strip().replace("\t", " ").split(" ")
		assert len(s) == 2
		utt, wav_file = s

		with wave.open(wav_file, 'r') as wr:
			data = wr.readframes(wr.getnframes())
			wav = np.frombuffer(data, dtype=np.int16)
			np.save("{}/{}".format(args.wav_outdir, utt), wav)
	wavlist_fp.close()

if __name__ == '__main__':
	main_()
