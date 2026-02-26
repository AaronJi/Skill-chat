#!/bin/bash

python inference.py --reverse True --topk 5 --url  http://10.209.76.23:8501/v1

if [ $? -eq 0 ]; then
    python inference.py --reverse True --url  http://10.209.76.23:8501/v1
else
    echo "script1.py fail!!! script2.py will not run."
    exit 1
fi