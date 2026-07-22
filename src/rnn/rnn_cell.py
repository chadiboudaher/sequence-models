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
        self.Woh = np.random.randn(output_size, hidden_size) * 0.01

        self.bh = np.zeros((hidden_size, 1))
        self.bo = np.zeros((output_size, 1))

    def forward(self, x, h_prev):
        self.h_raw = np.dot(self.Wxh, x) + np.dot(self.Whh, h_prev) + self.bh
        h = np.tanh(self.h_raw)

        y = np.dot(self.Woh, h) + self.bo

        return h, y

        
        