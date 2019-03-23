import os
import subprocess
import tempfile
import numpy as np
from .sentenceembedding import SentenceEmbeddingModel


class Laser(SentenceEmbeddingModel):
    """LASER Language-Agnostic SEntence Representations

    Calculates sentence embeddings using a pre-trained multilingual
    LASER model.

    References:

    Mikel Artetxe, Holger Schwenk: Massively Multilingual Sentence
    Embeddings for Zero-Shot Cross-Lingual Transfer and Beyond,
    https://arxiv.org/abs/1812.10464

    https://github.com/facebookresearch/LASER
    """
    
    def __init__(self, name, laser_path):
        super().__init__(name)
        self.laser_path = laser_path
        self.embedding_dim = 1024

    def describe(self):
        return '\n'.join([
            self.name,
            f'Embedding dimensionality: {self.embedding_dim}'
        ])

    def transform(self, sentences):
        input_filename = self.write_sentences_to_file(sentences)
        output_filename = self.generate_output_filename()

        try:
            self.run_laser_embeddings(input_filename, output_filename)
            return self.read_embeddings(output_filename)
        finally:
            os.unlink(output_filename)
            os.unlink(input_filename)

    def generate_output_filename(self):
        f, filename = tempfile.mkstemp('.raw', 'laser-sentence')
        os.close(f)
        return filename

    def write_sentences_to_file(self, sentences):
        f, filename = tempfile.mkstemp('.txt', 'laser-input')
        for s in sentences:
            os.write(f, s.encode('utf-8'))
            os.write(f, b'\n')
        os.close(f)
        return filename

    def run_laser_embeddings(self, input_filename, output_filename):
        args = [
            'bash',
            os.path.join(self.laser_path, 'tasks/embed/embed.sh'),
            input_filename,
            'fi',
            output_filename
        ]
        env = {'LASER': self.laser_path}

        subprocess.run(args, env=env, check=True)

    def read_embeddings(self, output_filename):
        X = np.fromfile(output_filename, dtype=np.float32, count=-1)
        X.resize(X.shape[0] // self.embedding_dim, self.embedding_dim)
        return X
