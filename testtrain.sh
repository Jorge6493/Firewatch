#!/bin/sh


python 0.1_Adam.py > logs/testtrain-0.1_Adam.txt
python 0.01_Adam.py > logs/testtrain-0.01_Adam.txt
python 0.001_Adam.py > logs/testtrain-0.001_Adam.txt
python 0.1_SGD.py > logs/testtrain-0.1_SGD.txt
python 0.01_SGD.py > logs/ltesttrain-0.01_SGD.txt
python 0.001_SGD.py > logs/testtrain-0.001_SGD.txt
python 0.0001_SGD.py > logs/testtrain-0.0001_SGD.txt
