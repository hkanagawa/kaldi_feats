#!/usr/bin/env python

import argparse
import numpy as np
from utils import kaldi_io
from utils.feat import FeatureOperator


def main_():
	ap = argparse.ArgumentParser(usage="")

	# positional args
	ap.add_argument("clip", help="varがclipとなるよう、二次統計量をいじる", type=float)
	ap.add_argument("dim_for_scaling", help="33-52,3 (same as select-feats)", type=str)
	ap.add_argument("stats_rspecifier", help="in", type=str)
	ap.add_argument("stats_wspecifier", help="out", type=str)

	args = ap.parse_args()

	dim_for_scaling = FeatureOperator.parse_dim_specifier(args.dim_for_scaling)
	clip = args.clip

	with kaldi_io.open_or_fd(args.stats_wspecifier, "wb") as stats_writer:
		for spk, rstats in kaldi_io.read_mat_ark(args.stats_rspecifier):
			T, D = rstats.shape
			assert (T == 2)

			count = rstats[0, -1]
			wstats = np.array(rstats)

			for d in dim_for_scaling:
				x, x2 = rstats[0, d], rstats[1, d]
				mean = x / count
				y = count * (mean * mean + clip)
				assert (y > 0)
				wstats[1, d] = y
				del x, x2, mean, y
			kaldi_io.write_mat(stats_writer, wstats, key=spk)


if __name__ == '__main__':
	main_()
