#!/bin/sh


python power_gen/src/train.py --fold 0 --model rf
python power_gen/src/train.py --fold 1 --model rf
python power_gen/src/train.py --fold 2 --model rf
python power_gen/src/train.py --fold 3 --model rf
python power_gen/src/train.py --fold 4 --model rf