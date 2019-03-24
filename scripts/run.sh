#!/bin/sh

export LASER=`pwd`/LASER

python -m fiSentenceEmbeddingEval.evaluate
python -m fiSentenceEmbeddingEval.plot_results
