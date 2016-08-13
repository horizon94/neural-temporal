#!/bin/bash

source $(dirname $0)/../../../../neural-assertion/scripts/keras/env/bin/activate

export PYTHONPATH=$PYTHONPATH:$(dirname $0)/../../../../apache-ctakes/ctakes-neural/scripts

subdir=`dirname $0`

python $(dirname $0)/timex_classify.py $* $subdir 

ret=$?

deactivate

exit $ret