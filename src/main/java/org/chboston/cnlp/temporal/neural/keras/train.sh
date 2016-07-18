#!/bin/bash

source $(dirname $0)/env/bin/activate

python $(dirname $0)/et_dimaCNN_train-and-package.py $*

ret=$?

deactivate

exit $ret
