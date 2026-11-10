import tempfile
import unittest

import torch

from pathlib import Path
from transformers import EsmConfig, EsmModel, EsmTokenizer

from biotrainer.embedding import (
    OneHotEncodingEmbedder,
    RandomEmbedder,
    get_embedding_service,
)
from biotrainer.embedding.huggingface import ESM2, HuggingfaceTransformerEmbedder

# Token order of the facebook/esm2_* vocab.txt
_ESM2_VOCAB = ["<cls>", "<pad>", "<eos>", "<unk>", *"LAGVSERTIDPKQNFYMHWCXBUZO.-", "<null_1>", "<mask>"]


def _tiny_esm_embedder(embedder_class):
    """ Randomly initialised tiny ESM-2 model with the real EsmTokenizer, to test without downloading weights """
    with tempfile.TemporaryDirectory() as tmp_dir_name:
        vocab_file = Path(tmp_dir_name) / "vocab.txt"
        vocab_file.write_text("\n".join(_ESM2_VOCAB))
        tokenizer = EsmTokenizer(vocab_file=str(vocab_file))
    torch.manual_seed(42)
    model = EsmModel(EsmConfig(vocab_size=len(_ESM2_VOCAB), pad_token_id=1, mask_token_id=32, hidden_size=16,
                               num_hidden_layers=2, num_attention_heads=2, intermediate_size=32,
                               position_embedding_type="rotary")).eval()
    return embedder_class(name="tiny_esm", model=model, tokenizer=tokenizer, use_half_precision=False,
                          device=torch.device("cpu"))


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

    def test_huggingface_embedder_removes_esm_special_tokens(self):
        """ Checks that the generic huggingface embedder removes the special tokens of the EsmTokenizer """
        embedder = _tiny_esm_embedder(HuggingfaceTransformerEmbedder)
        per_residue_embeddings = list(embedder.embed_many(self._test_sequences))
        for sequence, per_residue_embedding in zip(self._test_sequences, per_residue_embeddings):
            self.assertEqual((len(sequence), 16), tuple(per_residue_embedding.shape))

    def test_huggingface_embedder_keeps_mask_token(self):
        """ Checks that the generic huggingface embedder keeps the mask token, although it is a special token """
        embedder = _tiny_esm_embedder(HuggingfaceTransformerEmbedder)
        per_residue_embedding = next(embedder.embed_many(["SEQ<mask>NCE"]))
        self.assertEqual((7, 16), tuple(per_residue_embedding.shape))

    def test_attention_map_removes_esm_special_tokens(self):
        """ Checks that the attention map only covers the residues of the sequence """
        sequence = self._test_sequences[0]
        for embedder_class in [HuggingfaceTransformerEmbedder, ESM2]:
            with self.subTest(embedder=embedder_class.__name__):
                attention_map = _tiny_esm_embedder(embedder_class).compute_attention_map(sequence)
                # 2 layers x 2 heads
                self.assertEqual((len(sequence), len(sequence), 4), tuple(attention_map.shape))

    def test_attention_map_does_not_enable_tf32(self):
        # The TF32 flags are process-wide, enabling them would switch all later float32 computations to TF32
        matmul_tf32, cudnn_tf32 = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            _tiny_esm_embedder(ESM2).compute_attention_map(self._test_sequences[0])
            self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
            self.assertFalse(torch.backends.cudnn.allow_tf32)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
            torch.backends.cudnn.allow_tf32 = cudnn_tf32
