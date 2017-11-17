#!/bin/bash -f
#python3 CNN.py 128 40 1 1 model-60.h5
python3 CNN.py 512 100 0 5 None $1
#'batch', type=int, default=64
#'epoch', type=int, default=1
#'pretrain', type=int, default=0
#'save_every', type=int, default=1
#'model_name', type=str, default='model/model-1'
