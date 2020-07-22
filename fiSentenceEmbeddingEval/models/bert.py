import numpy as np
import transformers
from .sentenceembedding import SentenceEmbeddingModel


class Bert(SentenceEmbeddingModel):
    """BERT sentence embeddings

    Computes a sentence embedding vector as the average of last few
    layers of the special [CLS] token.

    Uses a pre-trained BERT model.
    """

    def __init__(self, name, transformer_name, layers=None):
        super().__init__(name)
        self.layers = [-1] if layers is None else layers
        self.tokenizer = transformers.BertTokenizer.from_pretrained(transformer_name)
        self.model = transformers.BertModel.from_pretrained(transformer_name)

    def describe(self):
        return '\n'.join([
            self.name,
            f'Embedding dimensionality: {self.model.config.hidden_size}',
            f'Vocabulary size: {self.model.config.vocab_size}',
            f'Layers: {", ".join(str(x) for x in self.layers)}'
        ])

    def transform(self, sentences):
        batch_size = 32
        embeddings = []
        for k in range(0, len(sentences), batch_size):
            sentence_batch = sentences[k:k+batch_size].tolist()
            tokens = self.tokenizer(sentence_batch, padding=True, return_tensors='pt')
            _, _, hidden_states = self.model(**tokens, output_hidden_states=True)

            num_sentences = hidden_states[0].shape[0]
            for i in range(num_sentences):
                layer_values = []
                for j in self.layers:
                    layer_values.append(hidden_states[j][i, 0, :].detach().numpy())
                X = np.atleast_2d(np.asarray(layer_values))
                embeddings.append(X.mean(axis=0))

        return np.asarray(embeddings)

    def transform_pairs(self, sentence_pairs):
        """Encode a pair of sentences as one vector"""
        return self.transform(sentence_pairs[['sentence1', 'sentence2']]
                              .to_records(index=False))
