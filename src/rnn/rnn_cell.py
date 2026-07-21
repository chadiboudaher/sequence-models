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

        self.U = np.random.randn(hidden_size, input_size) * 0.01
        self.W = np.random.randn(hidden_size, hidden_size) * 0.01
        self.V = np.random.randn(hidden_size, output_size) * 0.01

        self.b = np.zeros((hidden_size, 1))
        self.c = np.zeros((output_size, 1))

        def softmax(self, x):
            e = np.exp(x - np.max(x))
            return e / np.sum(e)

        def forward(self, x, h_prev):
            """
            Forward Pass of a single RNN step

            Args:
                x: Input of the current timestamp (input_size, 1)
                h_prev: previous hidden state (hidden_size, 1)

            Returns:
                h: Current hidden state
                y: Current output
            """

            self.a = self.b + np.dot(self.W, h_prev) + np.dot(self.U, x)
            h = np.tanh(self.a)

            y = self.c + np.dot(self.V, h)
            y = self.softmax(y)

            return h, y
        
        def backward(self, gradient_next, gradient_out, lr=0.01):
            """
            Backward Pass computes gradients and updates weights

            Args:
                gradient_next: Gradient from next step
                gradient_output: Gradient of output loss
            """
            ...