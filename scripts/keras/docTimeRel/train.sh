#!/bin/bash

source $(dirname $0)/../../../../ctakes/ctakes-temporal/scripts/keras/env/bin/activate

export PYTHONPATH=$PYTHONPATH:$(dirname $0)/../../../../ctakes/ctakes-neural/scripts

subdir=`dirname $0`

python $(dirname $0)/dimaCNN_train.py $* $subdir

ret=$?

deactivate

exit $ret