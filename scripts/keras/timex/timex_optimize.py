#!/usr/bin/env python

## python imports
import sys
import os, os.path
import pickle
import random

## library imports
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.wrappers import TimeDistributed
from keras.layers.embeddings import Embedding
from keras.layers import LSTM
from sklearn.cross_validation import train_test_split
from zipfile import ZipFile

## local imports
from random_search import RandomSearch
import cleartk_io as ctk_io
import nn_models as models


## Define parameter space for each hyperparameter
batch_sizes=(32, 64, 128, 256)
layers = ( (64,), (128,), (256,), (512,), (1024,) )
embed_dims = (25, 50, 100, 200)


def get_random_config():
    config = {}
    
    config['batch_size'] = random.choice(batch_sizes)
    config['layers'] = random.choice(layers)
    config['embed_dim'] = random.choice(embed_dims)
    
    return config

def run_one_eval(epochs, config, train_x, train_y, valid_x, valid_y, vocab_size, num_outputs):
    model = models.get_rnn_model(dimension=train_x.shape, vocab_size=vocab_size, num_outputs=num_outputs, layers=config['layers'], embed_dim=config['embed_dim'])
    
    history = model.fit(train_x,
            train_y,
            nb_epoch=epochs,
            batch_size=config['batch_size'],
            verbose=1,
            validation_data=(valid_x, valid_y))
    
    return history.history['loss'][-1]
    
def main(args):
    if len(args) < 1:
        sys.stderr.write("Error - two required arguments: <data directory>\n")
        sys.exit(-1)

    working_dir = args[0]
    
    (labels, label_alphabet, feats, feats_alphabet) = ctk_io.read_bio_sequence_data(working_dir)
    
    maxlen = max([len(seq) for seq in feats])
    all_x = pad_sequences(feats, maxlen=maxlen)
    all_y = pad_sequences(labels, maxlen=maxlen)
    
    train_x, valid_x, train_y, valid_y = train_test_split(all_x, all_y, test_size=0.2, random_state=18)
    
    optim = RandomSearch(lambda: get_random_config(), lambda x, y: run_one_eval(x, y, train_x, train_y, valid_x, valid_y, len(feats_alphabet), len(label_alphabet) ) )
    best_config = optim.optimize()
    
    model = models.get_rnn_model(all_x.shape, len(feats_alphabet), len(label_alphabet), layers=best_config['layers'], embed_dim=best_config['embed_dim'])

    model.fit(all_x,
            all_y,
            nb_epoch=epochs,
            batch_size=batch_size,
            verbose=1,
            validation_split=0.1)

    model.summary()
    
    json_string = model.to_json()
    open(os.path.join(working_dir, 'model_0.json'), 'w').write(json_string)
    model.save_weights(os.path.join(working_dir, 'model_0.h5'), overwrite=True)
    
    fn = open(os.path.join(working_dir, 'alphabets.pkl'), 'w')
    pickle.dump( (feats_alphabet, label_alphabet), fn)
    fn.close()

    with ZipFile(os.path.join(working_dir, 'script.model'), 'w') as myzip:
        myzip.write(os.path.join(working_dir, 'model_0.json'), 'model_0.json')
        myzip.write(os.path.join(working_dir, 'model_0.h5'), 'model_0.h5')
        myzip.write(os.path.join(working_dir, 'alphabets.pkl'), 'alphabets.pkl')

    
    
if __name__ == "__main__":
    main(sys.argv[1:])
