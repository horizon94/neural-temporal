#!/usr/bin/env python

import sys
import cleartk_io as ctk_io
import nn_models as models
import os, os.path
import pickle
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, model_from_json
from zipfile import ZipFile


def main(args):
    if len(args) < 1:
        sys.stderr.write("Error - two required arguments: <data directory>\n")
        sys.exit(-1)

    working_dir = args[0]

    with ZipFile(os.path.join(working_dir, 'script.model'), 'r') as myzip:
        myzip.extract('model_0.json', working_dir)
        myzip.extract('model_0.h5', working_dir)
        myzip.extract('alphabets.pkl', working_dir)

    (feature_alphabet, label_alphabet) = pickle.load( open(os.path.join(working_dir, 'alphabets.pkl'), 'r' ) )
    label_lookup = {val:key for (key,val) in label_alphabet.iteritems()}
    model = model_from_json(open(os.path.join(working_dir, "model_0.json")).read())
    model.load_weights(os.path.join(working_dir, "model_0.h5"))       
    
    input_seq_len = model.layers[0].input_shape[1]

    eos = False
    feats = []
    
    while True:
        try:
            line = sys.stdin.readline().strip()
            if not line:
                break
            
            feats = [ctk_io.read_bio_feats_with_alphabet(feat, feature_alphabet) for feat in line.split()]
            outputs = model.predict(np.array([feats]))[0]
            #output_labels = []
            pred_classes = [output.argmax() for output in outputs] 
            labels = [label_lookup[pred_class] for pred_class in pred_classes]
            ctk_io.print_label(' '.join(labels))
                #print("Output is %s\n%s" % ( str(outputs), label))
                #for ind in range(actual_len):
#                     pred_class = outputs[0][ind].argmax()
#                     label = label_lookup[pred_class]
#                     output_labels.append(label)
            
                #print("Output is %s, %s" % (str(outputs), output_labels))
                        
        except Exception as e:
            print("Exception thrown: %s" % (e))

if __name__ == "__main__":
    main(sys.argv[1:])
