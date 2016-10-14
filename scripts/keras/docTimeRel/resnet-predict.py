#!python

from keras.models import Sequential, model_from_json
import numpy as np
import cleartk_io as ctk_io
import sys
import os.path
import pickle
from keras.preprocessing.sequence import pad_sequences
from zipfile import ZipFile

def main(args):
    if len(args) < 1:
        sys.stderr.write("Error - one required argument: <model directory>\n")
        sys.exit(-1)

    working_dir = args[0]

    with ZipFile(os.path.join(working_dir, 'script.model'), 'r') as myzip:
        myzip.extract('model_0.json', working_dir)
        myzip.extract('model_0.h5', working_dir)
        myzip.extract('alphabets.pkl', working_dir)

    (feature_alphabet, label_alphabet, maxlen) = pickle.load( open(os.path.join(working_dir, 'alphabets.pkl'), 'r' ) )
    label_lookup = {val:key for (key,val) in label_alphabet.iteritems()}
    model = model_from_json(open(os.path.join(working_dir, "model_0.json")).read())
    model.load_weights(os.path.join(working_dir, "model_0.h5"))

    while True:
        try:
            line = sys.stdin.readline().rstrip()
            if not line:
                break

            ## Convert the line of Strings to lists of indices
            feats=[]
            for unigram in line.rstrip().split():
                if(feature_alphabet.has_key(unigram)):
                    feats.append(feature_alphabet[unigram])
                else:
                    feats.append(feature_alphabet["none"])
            if(len(feats)> maxlen):
                feats=feats[0:maxlen]
            test_x = pad_sequences([feats], maxlen=maxlen)
            #feats = np.reshape(feats, (1, 6, input_dims / 6))
            #feats = np.reshape(feats, (1, input_dims))

            X_dup = []
            X_dup.append(test_x)
            #X_dup.append(test_x)

            out = model.predict(X_dup)[0]

            out_str = label_lookup[out.argmax()]
            ctk_io.print_label(out_str)
            # print("Out is %s and decision is %d" % (out, out.argmax()))
        except Exception as e:
            print("Exception thrown: %s" % (e))



if __name__ == "__main__":
    main(sys.argv[1:])
