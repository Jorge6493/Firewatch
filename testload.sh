#!/bin/sh


python load2-0.1_Adam.py > logs/testload-0.1_Adam.txt
python load2-0.1_SGD.py > logs/testload-0.1_SGD.txt
python load2-0.01_SGD.py > logs/testload-0.01_SGD.txt

