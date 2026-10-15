import unittest

import torch

from biotrainer.embedding import (
    OneHotEncodingEmbedder,
    RandomEmbedder,
    get_embedding_service,
)


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

    def test_seeded_random_embedding_depends_only_on_seed_and_sequence(self):
        forward = list(RandomEmbedder(seed=1510).embed_many(["SEQWENCE", "PRTEIN"]))
        backward = list(RandomEmbedder(seed=1510).embed_many(["PRTEIN", "SEQWENCE"]))
        self.assertTrue(torch.equal(forward[0], backward[1]))
        self.assertTrue(torch.equal(forward[1], backward[0]))

    def test_random_embedding_changes_with_seed(self):
        first = next(RandomEmbedder(seed=1510).embed_many(["SEQWENCE"]))
        second = next(RandomEmbedder(seed=2903).embed_many(["SEQWENCE"]))
        self.assertFalse(torch.equal(first, second))

    def test_unseeded_random_embedder_stays_random(self):
        first = next(RandomEmbedder().embed_many(["SEQWENCE"]))
        second = next(RandomEmbedder().embed_many(["SEQWENCE"]))
        self.assertFalse(torch.equal(first, second))

    def test_embedding_service_seeds_the_random_embedder(self):
        def embed(seed):
            service = get_embedding_service(
                "random_embedder", custom_tokenizer_config=None, seed=seed
            )
            return next(
                service.generate_embeddings(input_data=["SEQWENCE"], reduce=True)
            ).embedding

        self.assertTrue(torch.equal(embed(1510), embed(1510)))
