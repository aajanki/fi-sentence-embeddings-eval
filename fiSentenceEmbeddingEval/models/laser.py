import codecs
import os
import os.path
import tempfile
import sys
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
    
    def __init__(self, name, laser_path, verbose=False):
        super().__init__(name)
        self.laser_path = laser_path
        self.embedding_dim = 1024
        self.verbose = verbose

    def describe(self):
        return '\n'.join([
            self.name,
            f'Embedding dimensionality: {self.embedding_dim}'
        ])

    def transform(self, sentences):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_filename = os.path.join(tmpdir, 'sentences.txt')
            with codecs.open(input_filename, 'w', 'utf-8') as inf:
                self.write_sentences_to_file(sentences, inf)

            output_filename = os.path.join(tmpdir, 'output.raw')

            self.laser_embed(input_filename, output_filename, tmpdir)
            return self.read_embeddings(output_filename)

    def write_sentences_to_file(self, sentences, f):
        for s in sentences:
            f.write(s)
            f.write('\n')

    def read_embeddings(self, output_filename):
        X = np.fromfile(output_filename, dtype=np.float32, count=-1)
        X.resize(X.shape[0] // self.embedding_dim, self.embedding_dim)
        return X

    def laser_embed(self, input_file, output_filename, tmpdir, token_lang='fi',
                    max_tokens=12000, buffer_size=10000):
        # This function is copied from LASER/source/embed.py
        #
        # LASER doesn't expose the main embedding code as a function,
        # so it needs to be copy pasted here.

        laser_source_path = os.path.join(self.laser_path, 'source')
        if laser_source_path not in sys.path:
            sys.path.append(laser_source_path)

        import embed
        from text_processing import Token, BPEfastApply

        model_dir = os.path.join(self.laser_path, 'models')
        encoder = os.path.join(model_dir, 'bilstm.93langs.2018-12-26.pt')
        bpe_codes = os.path.join(model_dir, '93langs.fcodes')

        encoder = embed.SentenceEncoder(encoder,
                                        max_sentences=None,
                                        max_tokens=max_tokens,
                                        sort_kind='quicksort',
                                        cpu=False)

        tok_fname = os.path.join(tmpdir, 'tok')
        Token(input_file,
              tok_fname,
              lang=token_lang,
              romanize=False,
              lower_case=True,
              gzip=False,
              verbose=self.verbose,
              over_write=False)
        ifname = tok_fname

        bpe_fname = os.path.join(tmpdir, 'bpe')
        BPEfastApply(ifname,
                     bpe_fname,
                     bpe_codes,
                     verbose=self.verbose,
                     over_write=False)
        ifname = bpe_fname

        embed.EncodeFile(encoder,
                         ifname,
                         output_filename,
                         verbose=self.verbose,
                         over_write=False,
                         buffer_size=buffer_size)
