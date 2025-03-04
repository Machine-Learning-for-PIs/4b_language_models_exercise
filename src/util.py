"""Utility code."""

import codecs
import collections
import os
import pickle

import numpy as np
import torch


class TextLoader:
    """Code to turn text files into training batches."""

    def __init__(self, data_dir, batch_size, seq_length, encoding="utf-8"):
        """Create the text loader."""
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.encoding = encoding

        input_file = os.path.join(data_dir, "input.txt")
        vocab_file = os.path.join(data_dir, "vocab.pkl")
        tensor_file = os.path.join(data_dir, "data.npy")

        if not (os.path.exists(vocab_file) and os.path.exists(tensor_file)):
            print("reading text file")
            self.preprocess(input_file, vocab_file, tensor_file)
        else:
            print("loading preprocessed files")
            self.load_preprocessed(vocab_file, tensor_file)
        self.create_batches()
        self.reset_batch_pointer()

    def preprocess(self, input_file, vocab_file, tensor_file):  # noqa: D102
        with codecs.open(input_file, "r", encoding=self.encoding) as f:
            data = f.read()
        counter = collections.Counter(data)
        count_pairs = sorted(counter.items(), key=lambda x: -x[1])
        self.chars, _ = zip(*count_pairs)
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        with open(vocab_file, "wb") as f:
            pickle.dump(self.chars, f)
        self.tensor = np.array(list(map(self.vocab.get, data)))
        np.save(tensor_file, self.tensor)

    def load_preprocessed(self, vocab_file, tensor_file):  # noqa: D102
        with open(vocab_file, "rb") as f:
            self.chars = pickle.load(f)
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.tensor = np.load(tensor_file)
        self.num_batches = int(self.tensor.size / (self.batch_size * self.seq_length))

    def create_batches(self):  # noqa: D102
        self.num_batches = int(self.tensor.size / (self.batch_size * self.seq_length))

        # When the data (tesor) is too small, let's give them a better error message
        if self.num_batches == 0:
            assert (  # noqa: B011
                False
            ), "Not enough data. Make seq_length and batch_size small."  # noqa: B011

        self.tensor = self.tensor[
            : self.num_batches * self.batch_size * self.seq_length
        ]
        xdata = self.tensor
        ydata = np.copy(self.tensor)
        ydata[:-1] = xdata[1:]
        ydata[-1] = xdata[0]
        self.x_batches = np.split(
            xdata.reshape(self.batch_size, -1), self.num_batches, 1
        )
        self.y_batches = np.split(
            ydata.reshape(self.batch_size, -1), self.num_batches, 1
        )

    def next_batch(self):  # noqa: D102
        x, y = self.x_batches[self.pointer], self.y_batches[self.pointer]
        self.pointer += 1
        return x, y

    def reset_batch_pointer(self):  # noqa: D102
        self.pointer = 0


def convert(sequences: torch.Tensor, inv_vocab: dict) -> list:
    """Convert an array of character-integers to a list of letters.

    Args:
        sequences (jnp.ndarray): An integer array, which represents characters.
        inv_vocab (dict): The dictonary with the integer to char mapping.

    Returns:
        list: A list of characters.
    """
    res = []
    # TODO: Return a nested list of characters.
    return res
