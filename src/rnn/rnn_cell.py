import torch
import numpy as np

class RNNCell:
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initialize RNN cell parameters

        Args:
            input_size: Dimension of input features.
            hidden_size: Dimension of hidden state.
            output_size: Dimension of output.
        """

        self.Wxh = np.random.randn(hidden_size, input_size) * 0.01
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.Woh = np.random.randn(hidden_size, output_size) * 0.01

        self.bt = np.zeros((hidden_size, 1))
        self.bo = np.zeros((output_size, 1))

        def softmax(self, x):
            e = np.exp(x - np.max(x))
            return e / np.sum(e)

        