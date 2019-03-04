#!/usr/bin/env python
# -*- coding: utf-8 -*-

from helper import *
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from generate import *
from LSTM import LSTM
from train import *

path = "Seinfeld_Scripts.txt"
text = load_data(path)

view_line_range = (0, 10)

print('Dataset Stats')
print('Roughly the number of unique words: {}'.format(len({word: None for word in text.split()})))

lines = text.split('\n')
print('Number of lines: {}'.format(len(lines)))
word_count_line = [len(line.split()) for line in lines]
print('Average number of words in each line: {}'.format(np.average(word_count_line)))

print()
print('The lines {} to {}:'.format(*view_line_range))
print('\n'.join(text.split('\n')[view_line_range[0]:view_line_range[1]]))

# preprocess and save data
preprocess_and_save_data(path, token_lookup, create_lookup_tables)

# load data
int_text, vocab_to_int, int_to_vocab, token_dict = load_preprocess()

# Check for a GPU
train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('No GPU found. Please use a GPU to train your neural network.')

# Data params
# Sequence Length
sequence_length = 10  # of words in a sequence
# Batch Size
batch_size = 128

# data loader - do not change
train_loader = batch_data(int_text, sequence_length, batch_size)
print(len(train_loader))
print(len(int_text))
print(len(vocab_to_int))

# Training parameters
# Number of Epochs
num_epochs = 1
# Learning Rate
learning_rate = 0.0001

# Model parameters
# Vocab size
vocab_size = len(vocab_to_int)
# Output size
output_size = len(vocab_to_int)
# Embedding Dimension
embedding_dim = 200
# Hidden Dimension
hidden_dim = 1024 #256
# Number of RNN Layers
n_layers = 2

# Show stats for every n number of batches
show_every_n_batches = 1000

# create model and move to gpu if available
rnn = LSTM.LSTM(vocab_size, output_size, embedding_dim, hidden_dim, n_layers, dropout=0.1)
print(rnn)
if train_on_gpu:
    rnn.cuda()

# defining loss and optimization functions for training
optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# training the model
trained_rnn = train_rnn(train_loader, rnn, batch_size, optimizer, criterion, num_epochs, show_every_n_batches)

# saving the trained model
helper.save_model('./save/trained_rnn', trained_rnn)
print('Model Trained and Saved')