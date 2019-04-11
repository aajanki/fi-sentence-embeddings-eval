#!/bin/sh

export LASER=`pwd`/LASER

python -m fiSentenceEmbeddingEval.evaluate --hyperparameters models/hyperparameters.json --num-trials 3
python -m fiSentenceEmbeddingEval.plot_results
