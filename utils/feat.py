from . import kaldi_io
import numpy as np

class FeatureOperator():
	@staticmethod
	def parse_dim_specifier(specifier: str) -> list:
		# convert string to int array, such as "2-5,10" --> [2, 3, 4, 5, 10].
		if specifier == "":
			return None
		dim = []
		tmp = specifier.split(",")
		for x in tmp:
			y = x.split("-")
			len_y = len(y)
			assert(len_y == 1 or len_y==2)

			if len_y == 1:
				dim.append(int(y[0]))
			else:
				[dim.append(i) for i in range(int(y[0]), int(y[1])+1)]
		return dim


	@staticmethod
	def read_next_feat(reader, key: str, num_frame: int, *, integer=False, length_tolerance=False) -> np.ndarray:
		assert (key == kaldi_io.read_key(reader))
		mat = kaldi_io.read_mat(reader) if integer is False else kaldi_io.read_vec_int(reader)
		if length_tolerance is False:
			assert (num_frame == mat.shape[0])
		return mat

	@staticmethod
	def convert_ali_to_1hot(ali: np.ndarray, target_dim: int) -> np.ndarray:
		num_frame = ali.shape[0]
		onehot = np.zeros((num_frame, target_dim))
		for t in range(num_frame):
			onehot[t, ali[t]] = 1
		return onehot

	@staticmethod
	def convert_1hot_to_conv2d_input(onehot: np.ndarray, target_dim: int) -> np.ndarray:
		num_frame, dim = onehot.shape

		ans = np.zeros((dim + 1, num_frame, target_dim))
		onevec = np.ones(target_dim)
		for t in range(num_frame):
			l_t = onehot[t, :]
			is_zero = (l_t == 0).all()
			ch = np.argmax(l_t) if is_zero else np.argmax(l_t) + 1
			ans[ch, t, :] = onevec
		return ans
