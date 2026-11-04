"""Contract tests for deriving variants from sequences."""

import unittest

from biotrainer_core.data_classes import Variant


class TestVariantFromSequences(unittest.TestCase):
    def test_descriptors(self):
        for variant, expected in [("SEQW", "V4W"), ("TEQV", "S1T"), ("TEQW", "S1T:V4W")]:
            with self.subTest(variant=variant):
                self.assertEqual(Variant.from_sequences("SEQV", variant).to_string(), expected)

    def test_from_sequence_is_inverse_of_get_mutant(self):
        reference = "SEQV"
        for variant in ["SEQW", "TEQW"]:
            with self.subTest(variant=variant):
                derived = Variant.from_sequences(reference, variant)
                self.assertEqual(derived.get_mutant_sequence(), variant)
                self.assertEqual(
                    Variant.parse(derived.to_string()).get_mutant_sequence(reference),
                    variant,
                )

    def test_length_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "insertions or deletions"):
            Variant.from_sequences("SEQV", "SEQ")

    def test_identical_sequences_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "must differ"):
            Variant.from_sequences("SEQV", "SEQV")


if __name__ == "__main__":
    unittest.main()
