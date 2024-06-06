import unittest

from biotrainer.embedders import OneHotEncodingEmbedder


class EmbeddersTests(unittest.TestCase):
    _test_sequences = ["SEQVENCEPRTEI",
                       "SEQWENCE",
                       "PRTEIN"]

    def test_embeddings_one_hot_encoding(self):
        embedder = OneHotEncodingEmbedder()

        # per residue embeddings
        per_residue_embeddings = list(embedder.embed_many(self._test_sequences))
        self.assertTrue(len(per_residue_embeddings) == len(self._test_sequences))

        for i, per_residue_embedding in enumerate(per_residue_embeddings):
            self.assertTrue(len(per_residue_embedding) == len(self._test_sequences[i]))
            self.assertTrue(all(
                len(residue_embedding) == embedder.embedding_dimension for residue_embedding in per_residue_embedding))

        # per sequence embeddings
        per_sequence_embeddings = [embedder.reduce_per_protein(embedding) for embedding in per_residue_embeddings]
        self.assertTrue(len(per_sequence_embeddings) == len(self._test_sequences))
        self.assertTrue(all([len(embedding) == embedder.embedding_dimension for embedding in per_sequence_embeddings]))
