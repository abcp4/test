import os
import numpy as np
import tensorflow as tf
from obspy.io.segy.segy import _read_segy

class SegyReader(object):

	def __init__(self, path, labels_path, batch_size, file_extension=".sgy", max_size=-1):
		"""
		:param path:
		:param labels_path:
		:param batch_size:
		:param file_extension:
		:param max_size:
		"""
		self._idx = None
		self.path = path
		self.labels_path = labels_path
		self._paths = list()
		self._labels = list()
		self._file_extension = file_extension
		self._max_size = max_size
		self._batch_size = batch_size
		self.load_data()

	def load_data(self):
		with open(self.labels_path, 'r') as f:
			for row in f:
				name, label = row.split(",")
				label = label.lower().strip()
				name = name.strip().replace(".png", ".segy")
				self._paths.append(os.path.join(self.path, name))
				if label == "good":
					self._labels.append(0)
				elif label == "bad":
					self._labels.append(1)
				elif label == "ugly":
					self._labels.append(2)
				else:
					raise ValueError("Label not recognized. Found in data: '{}'".format(label))
				if 0 < self._max_size == len(self._paths):
					break

		self._paths = np.asarray(self._paths)
		self._labels = np.asarray(self._labels)

	def load_from_dir(self):
		self._paths = list()
		self._labels = list()
		for root, _, files in os.walk(self.path):
			for file in files:
				if not file.endswith(self._file_extension):
					continue
				file_path = os.path.join(root, file)
				self._paths.append(file_path)
				self._labels.append(0)

				if 0 < self._max_size == len(self._paths):
					break
		self._paths = np.asarray(self._paths)
		self._labels = np.asarray(self._labels)

	def __len__(self):
		return len(self._paths)

	def __getitem__(self, item):
		if isinstance(item, int):
			return self._paths[item], self._labels[item]

		elif isinstance(item, slice):
			return self._paths[item], self._labels[item]

	def __iter__(self):
		"""
		Iterator initializer.
		"""
		self._idx = 0
		return self

	def __next__(self):
		"""
		Returns the iterator's next element.
		"""
		mod_batch = len(self) % self._batch_size
		if self._idx >= (len(self) - mod_batch):

			perm = np.random.permutation(len(self._paths))
			self._paths = self._paths[perm]
			self._labels = self._labels[perm]

			raise StopIteration()

		x = self.load_img(self._paths[self._idx])
		y = self._labels[self._idx]
		# index sum
		self._idx += 1
		return x, y

	def make_dataset(self):
		"""
		Returns a tensorflow Dataset created from this current iterator.
		:return: a `tf.data.Dataset`.
		"""
		batch_size = self._batch_size
		prefetch_buffer = 10
		dataset = tf.data.Dataset.from_generator(
			generator=lambda: iter(self),
			output_types=(tf.float32, tf.int32),
			# output_shapes=self._inputs_config["output_shapes"]
		)
		dataset = dataset.batch(batch_size)
		return dataset.prefetch(buffer_size=prefetch_buffer)

	def load_img(self, img_path):
		"""
		reads and normalize a seismogram from the given segy file.
		:param img_path: a path to the segy file.
		:return: seismogram image as numpy array normalized between 0-1.
		"""
		segy = _read_segy(img_path)
		_traces = list()
		for trace in segy.traces:
			_traces.append(trace.data)
		x = np.asarray(_traces, dtype=np.float32)
		std = x.std()
		x -= x.mean()
		x /= std
		x *= 0.1
		x += .5
		x = np.clip(x, 0, 1)

		return x.T
