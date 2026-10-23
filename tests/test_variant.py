"""Contract tests for deriving variants from sequences."""

import unittest

from biotrainer_core.data_classes import Variant


class TestVariantFromSequences(unittest.TestCase):
    def test_first_and_last_positions_are_one_based(self):
        self.assertEqual(Variant.from_sequences("ACDE", "VCDE").to_string(), "A1V")
        self.assertEqual(Variant.from_sequences("ACDE", "ACDA").to_string(), "E4A")

    def test_multiple_substitutions_are_colon_separated(self):
        self.assertEqual(Variant.from_sequences("ACDE", "VCDA").to_string(), "A1V:E4A")

    def test_round_trip_through_the_reference(self):
        reference = "ACDEFG"
        for variant in ["VCDEFG", "ACDAFG", "VCDAFG", "VWDAFA"]:
            with self.subTest(variant=variant):
                derived = Variant.from_sequences(reference, variant)
                self.assertEqual(derived.get_mutant_sequence(), variant)
                self.assertEqual(
                    Variant.parse(derived.to_string()).get_mutant_sequence(reference),
                    variant,
                )

    def test_reference_is_retained_for_reconstruction(self):
        derived = Variant.from_sequences("ACDE", "VCDA")
        self.assertEqual(derived.wt_sequence, "ACDE")
        self.assertEqual(derived.mutation_depth(), 2)

    def test_descriptors_do_not_depend_on_candidate_order(self):
        variants = ["VCDE", "ACDA"]
        forward = {v: Variant.from_sequences("ACDE", v).to_string() for v in variants}
        backward = {
            v: Variant.from_sequences("ACDE", v).to_string() for v in reversed(variants)
        }
        self.assertEqual(forward, backward)

    def test_invalid_variants_are_rejected(self):
        for reference, variant in [("ACDE", "ACDE"), ("ACDE", "ACD"), ("", "")]:
            with self.subTest(reference=reference, variant=variant):
                with self.assertRaises(ValueError):
                    Variant.from_sequences(reference, variant)


if __name__ == "__main__":
    unittest.main()
