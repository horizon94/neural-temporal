#!/usr/bin/env python

import sys
import cleartk_io as ctk_io
import nn_models as models
import os, os.path
import pickle
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.wrappers import TimeDistributed
from keras.layers.embeddings import Embedding
from keras.layers import LSTM

epochs=1
batch_size=64

def main(args):
    if len(args) < 2:
        sys.stderr.write("Error - two required arguments: <data directory> <alphabet directory>\n")
        sys.exit(-1)

    working_dir = args[0]
    script_dir = args[1]
    
    (labels, label_alphabet, feats, feats_alphabet) = ctk_io.read_bio_sequence_data(working_dir)
    
    maxlen = max([len(seq) for seq in feats])
    train_x = pad_sequences(feats, maxlen=maxlen)
    train_y = pad_sequences(labels, maxlen=maxlen)
    
    #print("After padding x has shape %s and y has shape %s" %  (str(train_x.shape), str(train_y.shape)))
    model = models.get_rnn_model(train_x.shape, len(feats_alphabet), len(label_alphabet))
    
    model.fit(train_x,
            train_y,
            nb_epoch=epochs,
            batch_size=batch_size,
            verbose=1,
            validation_split=0.1)

    model.summary()
    
    json_string = model.to_json()
    open(os.path.join(working_dir, 'model_0.json'), 'w').write(json_string)
    model.save_weights(os.path.join(working_dir, 'model_0.h5'), overwrite=True)
    
    fn = open(os.path.join(script_dir, 'alphabets.pkl'), 'w')
    pickle.dump( (feats_alphabet, label_alphabet), fn)
    fn.close()
    
if __name__ == "__main__":
    main(sys.argv[1:])
