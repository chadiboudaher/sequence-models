"""
Minimal Character-level  Vanilla RNN model
"""

import numpy as np

data = open("input.txt", "r").read()
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)

print(f"data has {data_size} characters, {vocab_size} unique.")

# Convert to integer before feeding it into a NN
char_to_ix = { ch:i for i, ch in enumerate(chars)}
ix_to_char = { i:ch for i, ch in enumerate(chars)}

# Hyperparameter
hidden_size = 1000 # size of hidden layer of neurons
seq_length = 25 # number of steps to unroll the RNN for
learning_rate = 1e-1

# model paramters
Wxh = np.random.randn(hidden_size, vocab_size) * 0.01
Whh = np.random.randn(hidden_size, hidden_size) * 0.01
Why = np.random.randn(vocab_size, hidden_size) * 0.01
bh = np.zeros((hidden_size, 1))
by = np.zeros((vocab_size, 1))