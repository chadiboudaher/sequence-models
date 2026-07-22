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

def lossFun(inputs, targets, hprev):
    """
    inputs, targets are both lists of integers
    hprev is Hx1 array of initial hidden state
    return loss, gradient on model parameters, and last hidden state
    """

    xs, hs, ys, ps = {}, {}, {}, {}

    hs[-1] = np.copy(hprev)
    loss = 0
    # Forward pass
    for t in range(len(inputs)):
        xs[t] = np.zeros((vocab_size, 1))
        xs[t][inputs[t]] = 1
        hs[t] = np.tanh(np.dot(Wxh, xs[t]) + 
                        np.dot(Whh, hs[t-1]) + 
                        bh)
        ys[t] = np.dot(Why, hs[t]) + by
        ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))
        loss += -np.log(ps[t][targets[t], 0])

    # Backward Pass
    dWxh, dWhh, dWhy = np.zeros_like(Wxh), 
    np.zeros_like(Whh), 
    np.zeros_like(Why)

    dbh, dby = np.zeros_like(bh), np.zeros_like(by)

    dhnext = np.zeros_like(hs[0])

    for t in reversed(range(len(inputs))):
        dy = np.copy(ps[t])
        dy[targets[t]] -= 1
        dWhy += np.dot(dy, hs[t].T)
        dby += dy

        dh = np.dot(Why.T, dy)
        dhraw = (1 - hs[t] * hs[t]) * dh
        dbh += dhraw

        dWxh += np.dot(dhraw, xs[t].T)
        dWhh += np.dot(dhraw, hs[t-1].T)
        dhnext = np.dot(Whh.T, dhraw)

    for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
        np.clip(dparam, -5, 5, out=dparam)
    return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1]

def sample(h, seed_ix, n):
    """
    sample a sequence of integers from the model
    h is memory state, seed_ix is seed letter for first time step
    """

    x = np.zeros((vocab_size, 1))
    x[seed_ix] = 1
    ixes = []

    for t in range(n):
        h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
        y = np.dot(Why, h) + by
        p = np.exp(y) / np.sum(np.exp(y))
        ix = np.random.choice(range(vocab_size), p=p.ravel())
        x = np.zeros((vocab_size, 1))
        x[ix] = 1
        ixes.append(ix)
    return ixes