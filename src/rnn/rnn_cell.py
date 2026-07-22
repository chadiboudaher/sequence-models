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
        """
        Initialise Forward Pass

        Args: 
            x: input at time stamp
            h_prev: Hidden state from previous time step

        Returns:
            h: Updated hidden state passed to the next time step.
            y: Output vector generated at the current time step.
        """
        self.x = x
        self.h_prev = h_prev

        self.h_raw = np.dot(self.Wxh, x) + np.dot(self.Whh, h_prev) + self.bh
        self.h = np.tanh(self.h_raw)

        self.y = np.dot(self.Woh, self.h) + self.bo

        return self.h, self.y
    
    def backward(self, dy,lr=0.01):
        """
        Backpropagation for a single cell

        Args:
        dy: gradient for loss function
        lr: learning rate used
        """
        dWoh = np.dot(dy, self.h.T)
        dbo = dy

        dh = np.dot(self.Woh.T, dy)

        dh_raw = dh * (1 - self.h ** 2)

        dWxh = np.dot(dh_raw, self.x.T)
        dWhh = np.dot(dh_raw, self.h_prev.T)
        dbh = dh_raw

        self.Woh -= lr * dWoh
        self.bo -= lr * dbo

        self.Wxh -= lr * dWxh
        self.Whh -= lr * dWhh
        self.bh -= lr * dbh