#!/bin/sh

export LASER=`pwd`/LASER

python -m fiSentenceEmbeddingEval.evaluate --hyperparameters models/hyperparameters.json
python -m fiSentenceEmbeddingEval.plot_results
