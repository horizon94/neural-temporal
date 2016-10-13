#!/usr/bin/env python

import numpy as np
np.random.seed(1339)
import sys
import cleartk_io as ctk_io
import nn_models as models
import os, os.path
import pickle
from keras.callbacks import EarlyStopping
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.wrappers import TimeDistributed
from keras.layers.embeddings import Embedding
from keras.layers import LSTM
from zipfile import ZipFile

from timex_common import get_model_for_config

epochs=20
batch_size=256
backwards=True
bilstm=True
layers=(64,)
embed_dim=25
best_config =  {'layers': (128,), 'backwards': True, 'bilstm': True, 'embed_dim': 100, 'activation': 'tanh', 'batch_size': 64, 'lr': 0.01, 'pretrain': False}

def main(args):
    if len(args) < 1:
        sys.stderr.write("Error - two required arguments: <data directory>\n")
        sys.exit(-1)

    working_dir = args[0]
    
    (labels, label_alphabet, feats, feats_alphabet) = ctk_io.read_bio_sequence_data(working_dir)

    weights = None
    if len(args) > 1 and best_config['pretrain'] == True:
        weights = ctk_io.read_embeddings(args[1], feats_alphabet)
    elif best_config['pretrain'] and len(args) == 1:
        sys.stderr.write("Error: Pretrain specified but no weights file given!")
        sys.exit(-1)
        
    maxlen = max([len(seq) for seq in feats])
    train_x = pad_sequences(feats, maxlen=maxlen)
    train_y = ctk_io.expand_labels(pad_sequences(labels, maxlen=maxlen), label_alphabet)
    
    #print("After padding x has shape %s and y has shape %s" %  (str(train_x.shape), str(train_y.shape)))
    model = get_model_for_config(train_x.shape, len(feats_alphabet), len(label_alphabet), best_config, weights)
    
    model.fit(train_x,
            train_y,
            nb_epoch=epochs,
            batch_size=best_config['batch_size'],
            verbose=1,
            validation_split=0.1,
            callbacks=[get_early_stopper()])

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

    
def get_early_stopper():
    return EarlyStopping(monitor='val_loss')

if __name__ == "__main__":
    main(sys.argv[1:])

