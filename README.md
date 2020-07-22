# tfsq: sequence modeling library for tensorflow

![ci](https://github.com/klknn/tfsq/workflows/ci/badge.svg)
[![codecov](https://codecov.io/gh/klknn/tfsq/branch/master/graph/badge.svg)](https://codecov.io/gh/klknn/tfsq)


reproducing Section 5 in https://arxiv.org/pdf/1308.0850.pdf

## requirements

- do `pip install -r requirements.txt`
- register your user/passwd at http://www.fki.inf.unibe.ch/databases/iam-on-line-handwriting-database

## TODO

- split net modules
- mixture-density layer
- eos layer
- biased sampling
- checkpoint https://www.tensorflow.org/api_docs/python/tf/compat/v1/train/Checkpoint
- tensorboard https://www.tensorflow.org/tensorboard/migrate
- parse xml in a robust way

## how to train

After setting up the requirements, try the all-in-one script:
```
python3 train.py --v=1 --num_epochs=100 --batch_size=12 --lr 1e-4 \
  --root ./data --http_user=... --http_password=...
```
It will download tgz (for the first time), preprocess data, and train neural networks.

### available options

TBD

## how to test

TBD

## LISENCE

BSL-1.0
