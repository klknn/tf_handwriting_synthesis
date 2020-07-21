# Handwriting synthesis in tensorflow

reproducing Section 5 in https://arxiv.org/pdf/1308.0850.pdf

## requirements

- tensorflow=2.2.0
- matplotlib=3.2.2
- register your user/passwd at http://www.fki.inf.unibe.ch/databases/iam-on-line-handwriting-database

## TODO

- attention layer
- lstm layer
- mixture-density layer
- eos layer
- biased sampling
- checkpoint https://www.tensorflow.org/api_docs/python/tf/compat/v1/train/Checkpoint
- tensorboard https://www.tensorflow.org/tensorboard/migrate

## how to train

After setting up the requirements, try the all-in-one script: 
```
./run.sh --db_user yourname --db_passwd yourpasswd
```

for downloading, preprocessing, training, and evaluating.

### available options

TBD

## how to test

TBD

## LISENCE

BSL-1.0
