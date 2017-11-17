#! /bin/bash -f
wget 'https://www.dropbox.com/s/5p6xqocxjg6nm7q/model-100.h5?dl=1'
python3 prediction.py $1 $2