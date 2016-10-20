#!/usr/bin/env python

import sklearn as sk

import numpy as np
np.random.seed(1337)

import cleartk_io as ctk_io
import nn_models

import sys
import os.path

import dataset

import keras as k
from keras.utils.np_utils import to_categorical
from keras.optimizers import RMSprop
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Merge
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers.embeddings import Embedding

from keras.layers.normalization import BatchNormalization

import pickle
from zipfile import ZipFile

# Helper to build a BN -> relu -> conv block
# This is an improved scheme proposed in http://arxiv.org/pdf/1603.05027v2.pdf
def _bn_relu_conv(nb_filter, subsample=1):
    def f(input):
        norm = BatchNormalization(mode=0, axis=1)(input)
        activation = Activation("relu")(norm)
        return Convolution1D(nb_filter=nb_filter, subsample_length=subsample,
                             init="he_normal", border_mode="same")(activation)

    return f

def main(args):
    if len(args) < 1:
        sys.stderr.write("Error - one required argument: <data directory>\n")
        sys.exit(-1)

    working_dir = args[0]

    (train_y, label_alphabet, train_x, feats_alphabet) = ctk_io.read_token_sequence_data(working_dir)


    init_vectors = None #used for pre-trained embeddings

    #load embeddings file
    embedingFile = '/Users/chenlin/Programming/ctakesWorkspace/neural-temporal/src/main/resources/org/apache/ctakes/temporal/thyme_word2vec_timex_50.vec'
    weights = ctk_io.read_embeddings(embedingFile, feats_alphabet)
    # if len(args) > 1 and best_config['pretrain'] == True:
    #     weights = ctk_io.read_embeddings(args[1], feats_alphabet)
    # elif best_config['pretrain'] and len(args) == 1:
    #     sys.stderr.write("Error: Pretrain specified but no weights file given!")
    #     sys.exit(-1)
    
    # turn x and y into numpy array among other things
    maxlen = max([len(seq) for seq in train_x])
    outcomes = set(train_y)
    classes = len(outcomes)

    train_x = pad_sequences(train_x, maxlen=maxlen)
    train_y = to_categorical(np.array(train_y), classes)

    #pickle.dump(maxlen, open(os.path.join(working_dir, 'maxlen.p'),"wb"))
    #pickle.dump(dataset1.alphabet, open(os.path.join(working_dir, 'alphabet.p'),"wb"))
    #test_x = pad_sequences(test_x, maxlen=maxlen)
    #test_y = to_categorical(np.array(test_y), classes)

    print 'train_x shape:', train_x.shape
    print 'train_y shape:', train_y.shape

    branches = [] # models to be merged
    train_xs = [] # train x for each branch
    #test_xs = []  # test x for each branch

    filtlens = "1,2,3,4,5"
    for filter_len in filtlens.split(','):
        branch = Sequential()
        branch.add(Embedding(len(feats_alphabet),
                         weights.shape[1],
                         input_length=maxlen,
                         weights=[weights],
                         trainable = False))
        branch.add(Convolution1D(nb_filter=200,
                             filter_length=int(filter_len),
                             border_mode='valid',
                             activation='relu',
                             subsample_length=1))
        branch.add(MaxPooling1D(pool_length=2))
        branch.add(Flatten())

        branches.append(branch)
        train_xs.append(train_x)
        #test_xs.append(test_x)
    branch = Sequential()
    branch.add(Embedding(len(feats_alphabet),
                         weights.shape[1],
                         input_length=maxlen,
                         weights=[weights],
                         trainable = False))
    branch.add(Convolution1D(nb_filter=200,
                             filter_length=3,
                             border_mode='valid',
                             activation='relu',
                             subsample_length=1))
    branch.add(Convolution1D(nb_filter=200,
                             filter_length=3,
                             border_mode='same',
                             activation='relu',
                             subsample_length=1))
    branch.add(MaxPooling1D(pool_length=2))
    branch.add(Flatten())

    branches.append(branch)
    train_xs.append(train_x)

    model = Sequential()
    model.add(Merge(branches, mode='concat'))

    model.add(Dense(250))#cfg.getint('cnn', 'hidden')))
    model.add(Dropout(0.25))#cfg.getfloat('cnn', 'dropout')))
    model.add(Activation('relu'))

    model.add(Dropout(0.25))#cfg.getfloat('cnn', 'dropout')))
    model.add(Dense(classes))
    model.add(Activation('softmax'))

    optimizer = RMSprop(lr=0.0001,#cfg.getfloat('cnn', 'learnrt'),
                      rho=0.9, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy',
                optimizer=optimizer,
                metrics=['accuracy'])
    model.fit(train_xs,
            train_y,
            nb_epoch=20,#cfg.getint('cnn', 'epochs'),
            batch_size=50,#cfg.getint('cnn', 'batches'),
            verbose=1,
            validation_split=0.1,
            class_weight=None)

    model.summary()

    json_string = model.to_json()
    open(os.path.join(working_dir, 'model_0.json'), 'w').write(json_string)
    model.save_weights(os.path.join(working_dir, 'model_0.h5'), overwrite=True)

    fn = open(os.path.join(working_dir, 'alphabets.pkl'), 'w')
    pickle.dump( (feats_alphabet, label_alphabet, maxlen), fn)
    fn.close()

    with ZipFile(os.path.join(working_dir, 'script.model'), 'w') as myzip:
        myzip.write(os.path.join(working_dir, 'model_0.json'), 'model_0.json')
        myzip.write(os.path.join(working_dir, 'model_0.h5'), 'model_0.h5')
        myzip.write(os.path.join(working_dir, 'alphabets.pkl'), 'alphabets.pkl')

    sys.exit(0)

if __name__ == "__main__":
    main(sys.argv[1:])