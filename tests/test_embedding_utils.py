import unittest

from biotrainer.embedders.interfaces import preprocess_sequences_with_whitespaces, preprocess_sequences_without_whitespaces


class EmbeddingUtilsTests(unittest.TestCase):

    def test_preprocessing_whitespace(self):
        mask_tokens = ["[MASK]", "<extra_id_0>"]

        # Basic cases without tokens
        sequences_without_token = [
            "PRTEINS",
            "S",  # Single amino acid
            "QESWENSCE",
            "SEQS",
            "UZOB",  # Special amino acids that should be converted to X
            "PROTEIN" * 10,  # Long sequence
        ]

        # Edge cases with tokens
        edge_cases = [
            "{token}",  # Only token
            "{token}{token}",  # Multiple consecutive tokens
            "A{token}",  # Token at end
            "{token}A",  # Token at start
            "A{token}A{token}A",  # Multiple tokens with single amino acids
            " {token} ",  # Spaces around token (should be cleaned)
            "UZO{token}B",  # Special amino acids with token
        ]

        # Test sequences without tokens
        for mask_token in mask_tokens:
            preprocessed = preprocess_sequences_with_whitespaces(sequences_without_token, mask_token)
            preprocessed_with_none = preprocess_sequences_with_whitespaces(sequences_without_token, None)

            for sequence in preprocessed + preprocessed_with_none:
                # Basic checks
                self.assertTrue(sequence[-1] != " ")
                self.assertTrue(mask_token not in sequence)

                # Check spacing
                self.assertEqual(len(sequence.split()), len(sequence.replace(" ", "")))

                # Check special amino acid conversion
                self.assertFalse(any(aa in sequence for aa in "UZOB"))

                # Check no double spaces
                self.assertFalse("  " in sequence)

        # Test sequences with tokens
        for mask_token in mask_tokens:
            # Format edge cases with actual token
            test_sequences = [case.format(token=mask_token) for case in edge_cases]

            # Add more complex cases
            test_sequences.extend([
                f"PROTEIN{mask_token}SEQUENCE",
                f"{mask_token}".join(["A"] * 5),  # Multiple tokens
                f"PRO{mask_token}TE{mask_token}IN",
                f"{mask_token}PROTEIN{mask_token}",
                f"A{mask_token}" * 5,  # Repeating pattern
            ])

            preprocessed = preprocess_sequences_with_whitespaces(test_sequences, mask_token)

            for orig, processed in zip(test_sequences, preprocessed):
                # Token preservation
                self.assertEqual(
                    orig.count(mask_token),
                    processed.count(mask_token),
                    f"Token count mismatch in {orig} -> {processed}"
                )

                # Check token integrity
                token_parts = processed.split(mask_token)
                for part in token_parts:
                    if part:  # Skip empty parts
                        # Each amino acid should be space-separated
                        self.assertEqual(
                            len(part.strip().split()),
                            len(part.strip().replace(" ", "")),
                            f"Incorrect spacing in part: {part}"
                        )

                # Check special amino acid conversion
                self.assertFalse(any(aa in processed for aa in "UZOB"))

                # Check no trailing/leading spaces
                self.assertEqual(processed, processed.strip())

        # Check that tokens in the middle of the sequence are correctly handled e.g. T[MASK]P -> T [MASK] P
        for mask_token in mask_tokens:
            preprocessed = preprocess_sequences_with_whitespaces([f"T{mask_token}P"], mask_token=mask_token)
            for processed in preprocessed:
                self.assertTrue(f" {mask_token} " in processed)


    def test_preprocessing_without_whitespace(self):
        sequences_without_token = [
            "PRTEINS",
            "S",  # Single amino acid
            "QESWENSCE",
            "SEQS",
            "UZOB",  # Special amino acids that should be converted to X
            "PROTEIN" * 10,  # Long sequence
        ]
        preprocessed = preprocess_sequences_without_whitespaces(sequences_without_token, mask_token=None)

        for processed in preprocessed:
            self.assertFalse(" " in processed)
            self.assertEqual(processed, processed.strip())
            self.assertFalse(any(aa in processed for aa in "UZOB"))
