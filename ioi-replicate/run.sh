#!/bin/bash

pip install git+https://github.com/redwoodresearch/Easy-Transformer.git && \
pip install einops datasets transformers fancy_einsum && \

python ioi.py > ./logs/run.txt
