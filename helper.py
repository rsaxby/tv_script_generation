#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pickle
import torch
from collections import Counter


SPECIAL_WORDS = {'PADDING': '<PAD>'}

def create_lookup_tables(text):
    """
    Create lookup tables for vocabulary
    :param text: The text of tv scripts split into words
    :return: A tuple of dicts (vocab_to_int, int_to_vocab)
    """
    word_counts = Counter(text)
    # sorting the words from most to least frequent in text occurrence
    sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
    vocab_to_int = dict((word, num) for num, word in enumerate(sorted_vocab))
    int_to_vocab = dict((num, word) for word, num  in vocab_to_int.items())
    # return tuple
    return (vocab_to_int, int_to_vocab)

def token_lookup():
    """
    Generate a dict to turn punctuation into a token.
    :return: Tokenized dictionary where the key is the punctuation and the value is the token
    """
    tokenized_punctuation= {'.': '<PERIOD>'}
    tokenized_punctuation[',']= '<COMMA>'
    tokenized_punctuation['"']= '<QUOTATION_MARK>'
    tokenized_punctuation[';']= '<SEMICOLON>'
    tokenized_punctuation['!']= '<EXCLAMATION_MARK>'
    tokenized_punctuation['?']= '<QUESTION_MARK>'
    tokenized_punctuation['(']= '<LEFT_PAREN>'
    tokenized_punctuation[')']= '<RIGHT_PAREN>'
    # tokenized_punctuation['--']= '<HYPHENS>'
    tokenized_punctuation['?']= '<QUESTION_MARK>'
    # tokenized_punctuation[':']= '<COLON>'  
    tokenized_punctuation['*']= '<ASTERISK>'
    tokenized_punctuation['-']= '<DASH>'
    tokenized_punctuation['\n']= '<NEWLINE>'
    return tokenized_punctuation


def batch_data(words, sequence_length, batch_size):
    """
    Batch the neural network data using DataLoader
    Args:
      words (list): int-encoded words from the tv scripts
      sequence_length (int): sequence length of each batch
      batch_size (int): size of each batch; the number of sequences in a batch
    Returns:
      DataLoader: dataloader with batched data
    """
    n_batches = len(words)//batch_size
    # only full batches
    words = words[:n_batches*batch_size]
    target_idx = len(words) - sequence_length
    x, y = [], []
    for idx in range(0, target_idx):
        end = idx+sequence_length
        batch_x = words[idx:end]
        x.append(batch_x)
        batch_y = words[end]
        y.append(batch_y)

    dataset = torch.utils.data.TensorDataset(torch.tensor(x), torch.tensor(y))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # return a dataloader
    return dataloader

def load_data(path):
    """
    Load Dataset from File
    """
    input_file = os.path.join(path)
    with open(input_file, "r") as f:
        data = f.read()

    return data


def preprocess_and_save_data(dataset_path, token_lookup, create_lookup_tables):
    """
    Preprocess Text Data
    """
    text = load_data(dataset_path)
    
    token_dict = token_lookup()
    for key, token in token_dict.items():
        text = text.replace(key, ' {} '.format(token))

    text = text.lower()
    text = text.split()

    vocab_to_int, int_to_vocab = create_lookup_tables(text + list(SPECIAL_WORDS.values()))
    int_text = [vocab_to_int[word] for word in text]
    pickle.dump((int_text, vocab_to_int, int_to_vocab, token_dict), open('preprocess.p', 'wb'))


def load_preprocess():
    """
    Load the Preprocessed Training data and return them in batches of <batch_size> or less
    """
    return pickle.load(open('preprocess.p', mode='rb'))


def save_model(filename, model):
    """ 
    Save the model.
    Args:
        filename (str): filname for the model
        model: model to be saved
    """
    save_filename = os.path.splitext(os.path.basename(filename))[0] + '.pt'
    torch.save(model, save_filename)


def load_model(filename):
    save_filename = os.path.splitext(os.path.basename(filename))[0] + '.pt'
    return torch.load(save_filename)

def save_text(script, file_name):
    # save script to a text file
    f =  open(file_name,"w")
    f.write(generated_script)
    f.close()
