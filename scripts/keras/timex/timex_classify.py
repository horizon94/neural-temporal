#!/usr/bin/env python

import sys
import cleartk_io as ctk_io
import nn_models as models
import os, os.path
import pickle
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, model_from_json


def main(args):
    if len(args) < 2:
        sys.stderr.write("Error - two required arguments: <data directory> <alphabet directory>\n")
        sys.exit(-1)

    working_dir = args[0]
    script_dir = args[1]

    (feature_alphabet, label_alphabet) = pickle.load( open(os.path.join(script_dir, 'alphabets.pkl'), 'r' ) )
    label_lookup = {val:key for (key,val) in label_alphabet.iteritems()}
    model = model_from_json(open(os.path.join(working_dir, "model_0.json")).read())
    model.load_weights(os.path.join(working_dir, "model_0.h5"))       
    
    input_seq_len = model.layers[0].input_shape[1]

    eos = False
    feats = []
    
    while True:
        try:
            line = sys.stdin.readline().rstrip()
            if not line:
                break
            
            if line == 'EOS':
                actual_len = len(feats)
                padded_feats = pad_sequences( [feats], input_seq_len)
                outputs = model.predict(np.array(padded_feats))
                output_labels = []
                
                for ind in range(actual_len):
                    pred_class = outputs[0][ind].argmax()
                    label = label_lookup[pred_class]
                    output_labels.append(label)
                    
                print("Output is %s, %s" % (str(outputs), output_labels))
            else:
                feats.append(ctk_io.read_bio_feat_with_alphabet(line, feature_alphabet))
        
        except Exception as e:
            print("Exception thrown: %s" % (e))

if __name__ == "__main__":
    main(sys.argv[1:])
