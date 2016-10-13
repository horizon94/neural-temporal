#!/usr/bin/env python

import numpy as np
np.random.seed(1338)

## python imports
import sys
import os, os.path
import pickle
import random

## library imports
from keras.callbacks import EarlyStopping
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.wrappers import TimeDistributed
from keras.layers.embeddings import Embedding
from keras.layers import LSTM

from sklearn.model_selection import train_test_split
from zipfile import ZipFile

## local imports
from random_search import RandomSearch
import cleartk_io as ctk_io
import nn_models as models
from timex_common import get_model_for_config

## Define parameter space for each hyperparameter
batch_sizes=(64, 128, 256)
layers = ( (64,), (128,), (256,), (512,), (1024,) )
embed_dims = (25, 50, 100, 201)
activations = ('tanh',)
backwards = (True, False)
bilstm = (True, False)
lrs = (0.1, 0.01, 0.001)
pretrain = (True, False)

def get_random_config(weights=None):
    config = {}
    
    config['batch_size'] = random.choice(batch_sizes)
    config['layers'] = random.choice(layers)
    if not weights is None:
        config['pretrain'] = random.choice(pretrain)
        if config['pretrain']:
            config['embed_dim'] = weights.shape[1]
        else:
            config['embed_dim'] = random.choice(embed_dims)
    else:
        config['embed_dim'] = random.choice(embed_dims)
        
    config['activation'] = random.choice(activations)
    config['backwards'] = random.choice(backwards)
    config['bilstm'] = random.choice(bilstm)
    config['lr'] = random.choice(lrs)
    
    return config

def run_one_eval(epochs, config, train_x, train_y, valid_x, valid_y, vocab_size, num_outputs, weights):

    print("Running with config: %s"  % str(config) )
    if not config['pretrain']:
        weights = None
    
    if config['pretrain'] and weights is None:
        raise Exception("ERROR: Pretrain flag given but no weights passed in!")
        
    model = get_model_for_config(train_x.shape, vocab_size, num_outputs, config, weights)
    
    history = model.fit(train_x,
            train_y,
            nb_epoch=epochs,
            batch_size=config['batch_size'],
            verbose=1,
            validation_data=(valid_x, valid_y), callbacks=[get_early_stopper()])
    
    #pred_y = model.predict(valid_x)
    
    ## Represent the quality of this configuration with loss on the validation set of the last (-1th) epoch
    for i in range(len(history.epoch)):
        loss = history.history['val_loss'][-1-i]
        if not np.isnan(loss):
            break
    
    print("Returning loss %f " % (loss) )
    return loss

def get_early_stopper():
    return EarlyStopping(monitor='val_loss')

def main(args):
    
    if len(args) < 1:
        sys.stderr.write("Error - one required argument: <data directory> [(optional) weights file]\n")
        sys.exit(-1)

    working_dir = args[0]
    
    (labels, label_alphabet, feats, feats_alphabet) = ctk_io.read_bio_sequence_data(working_dir)
    
    weights = None
    if len(args) > 1:
        weights = ctk_io.read_embeddings(args[1], feats_alphabet)
        
    maxlen = max([len(seq) for seq in feats])
    all_x = pad_sequences(feats, maxlen=maxlen)
    all_y = ctk_io.expand_labels(pad_sequences(labels, maxlen=maxlen), label_alphabet)

    train_x, valid_x, train_y, valid_y = train_test_split(all_x, all_y, test_size=0.2, random_state=7)
    
    optim = RandomSearch(lambda: get_random_config(weights), lambda x, y: run_one_eval(x, y, train_x, train_y, valid_x, valid_y, len(feats_alphabet), len(label_alphabet), weights ) )
    best_config = optim.optimize()
    
    open(os.path.join(working_dir, 'model_0.config'), 'w').write( str(best_config) )
    print("Best config returned by optimizer is %s" % str(best_config) )
    
    if not best_config['pretrain']:
        weights = None
        
    model = get_model_for_config(train_x.shape, len(feats_alphabet), len(label_alphabet), best_config, weights=weights)

    model.fit(all_x,
            all_y,
            nb_epoch=40,
            batch_size=best_config['batch_size'],
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
