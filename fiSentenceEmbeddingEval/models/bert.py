import json
import numpy as np
import os
import os.path
import tempfile
import bert.extract_features


class Bert:
    """BERT sentence embeddings

    Computes a sentence embedding vector as the average of last few
    layers (num_layers) of the special [CLS] token.

    Uses a pre-trained BERT model.
    """

    def __init__(self, path, num_layers=1):
        self.name = 'Bert multilingual'
        self.path = path
        self.num_layers = num_layers

        config = json.load(open(os.path.join(self.path, 'bert_config.json')))
        self.vocab_size = config.get('vocab_size')
        self.embedding_dim = config.get('pooler_fc_size')

    def fit(self, sentences):
        # nothing to do
        pass

    def describe(self):
        return '\n'.join([
            f'Embedding dimensionality: {self.embedding_dim}',
            f'Vocabulary size: {self.vocab_size}',
            f'Number of combined layers: {self.num_layers}'
        ])

    def transform(self, sentences):
        layers = ','.join(f'-{i+1}' for i in range(self.num_layers))
        
        out_f, output_filename = tempfile.mkstemp('.jsonl', 'bert-output')
        os.close(out_f)

        in_f, input_filename = tempfile.mkstemp('.txt', 'bert-input')
        for s in sentences:
            os.write(in_f, s.encode('utf-8'))
            os.write(in_f, b'\n')
        os.close(in_f)

        try:
            args = [
                '--input_file=' + input_filename,
                '--output_file=' + output_filename,
                '--vocab_file=' + os.path.join(self.path, 'vocab.txt'),
                '--bert_config_file=' + os.path.join(self.path, 'bert_config.json'),
                '--init_checkpoint=' + os.path.join(self.path, 'bert_model.ckpt'),
                '--layers=' + layers,
                '--max_seq_length=128',
                '--batch_size=8'
            ]

            flags_passthrough = bert.extract_features.FLAGS(
                ['extract_features.py'] + args,
                known_only=True)
            bert.extract_features.main(flags_passthrough)

            embeddings = []
            for line in open(output_filename).readlines():
                layers = json.loads(line).get('features')[0].get('layers')
                values = [x.get('values') for x in layers]
                values_matrix = np.atleast_2d(np.asarray(values))
                embeddings.append(values_matrix.mean(axis=0))

            return np.asarray(embeddings)
        finally:
            os.unlink(output_filename)
            os.unlink(input_filename)
