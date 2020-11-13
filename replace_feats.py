#!/usr/bin/env python
from utils import kaldi_io
from utils.feat import FeatureOperator as FO
import numpy as np
import argparse

def main_():
	ap = argparse.ArgumentParser()
	ap.add_argument("feat_src_rspecifier", type=str)
	ap.add_argument("feat_tgt_rspecifier", type=str)
	ap.add_argument("tgt_start_dim", type=int)
	ap.add_argument("feat_wspecifier", type=str)

	args = ap.parse_args()
	tgt_start_dim = args.tgt_start_dim
	with kaldi_io.open_or_fd(args.feat_wspecifier, mode="wb") as feat_writer:
		with kaldi_io.open_or_fd(args.feat_tgt_rspecifier, mode="rb") as feat_tgt_reader:
			for utt, feat_src in kaldi_io.read_mat_ark(args.feat_src_rspecifier):
				new_feat = feat_src.copy()
				T1, _ = new_feat.shape

				feat_tgt = FO.read_next_feat(feat_tgt_reader, key=utt, num_frame=T1)
				_, D2 = feat_tgt.shape
				new_feat[:, tgt_start_dim:tgt_start_dim+D2] = feat_tgt
				kaldi_io.write_mat(feat_writer, new_feat, key=utt)


if __name__ == '__main__':
	main_()
