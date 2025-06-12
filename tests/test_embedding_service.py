import os
import psutil
import random
import unittest
import tempfile

from pathlib import Path

from biotrainer.input_files import BiotrainerSequenceRecord
from biotrainer.protocols import Protocol
from biotrainer.embedders import OneHotEncodingEmbedder, EmbeddingService


class TestEmbeddingService(unittest.TestCase):
    def setUp(self):
        self._setup_test_environment()
        self._setup_fasta_parameters()

    def _setup_test_environment(self):
        print('Setting up test environment')
        self.embedder = OneHotEncodingEmbedder()
        self.embedding_service = EmbeddingService(embedder=self.embedder)

    def _setup_fasta_parameters(self):
        self.num_reads = 100
        if False:
            # Test with large sequences disabled - It has been tested throughout and should only be activated again
            # if problems arise, because not doing this test hugely speeds up the CI pipeline
            print("Generating ultra-long sequences to test memory limits and extreme edge cases.")
            print(
                "Sequence generation may take considerable time due to large memory allocation: "
                f"{psutil.virtual_memory().available / (1024 ** 3):.2f} GB available."
            )
            print(
                "The calculations for embeddings of these ultra-long sequences are also computationally intensive, "
                "which contributes to the overall long duration of the test."
            )
            # Use the original calculation in CI environment
            self.long_length = int((0.75 * psutil.virtual_memory().available) / (18 * 21))
        else:
            # Use a fixed value for local development
            self.long_length = 50000
        self.other_length = 250

    @staticmethod
    def _generate_sequence(length):
        """Generate a random sequence of a given length."""
        return ''.join(random.choice("ACDE") for _ in range(length))

    def _generate_fasta(self, num_reads, long_length, other_length, filename, include_long=True, include_short=True):
        """Generate a FASTA file with a specified number of reads."""
        print(f"Generating FASTA file: {filename}")
        with open(filename, 'w') as file:
            if include_long:
                print(f"Generating long sequence, it may take a bit of time.")
                for i in range(1, 3):
                    file.write(f">read_{i}\n{self._generate_sequence(long_length)}\n")

            if include_short:
                for i in range(3, num_reads + 1):
                    file.write(f">read_{i}\n{self._generate_sequence(other_length)}\n")
        print(f"FASTA file generated: {filename}")

    def _run_embedding_test(self, test_name, num_reads, protocol, include_long=True, include_short=True):
        with tempfile.TemporaryDirectory() as tmp_dir:
            print(f'Starting test: compute_embeddings_{test_name}')
            sequence_path = os.path.join(tmp_dir, f"{test_name}.fasta")
            self._generate_fasta(num_reads, self.long_length, self.other_length, sequence_path, include_long,
                                 include_short)
            result = self._compute_embeddings(sequence_path, tmp_dir, protocol)
            self._verify_result(protocol, result, tmp_dir)
            print(f'Test compute_embeddings_{test_name} completed successfully')

    def _compute_embeddings(self, sequence_path, output_dir, protocol):
        print('Computing embeddings')
        return self.embedding_service.compute_embeddings(
            sequence_path,
            Path(output_dir),
            protocol
        )

    def _verify_result(self, protocol, result, tmp_dir):
        self.assertTrue(os.path.exists(result), f"Result file does not exist: {result}")
        if protocol == Protocol.sequence_to_class:
            expected_path = os.path.join(tmp_dir, "sequence_to_class", "one_hot_encoding",
                                         "reduced_embeddings_file_one_hot_encoding.h5")
        elif protocol == Protocol.residue_to_class:
            expected_path = os.path.join(tmp_dir, "residue_to_class", "one_hot_encoding",
                                         "embeddings_file_one_hot_encoding.h5")
        self.assertEqual(result, expected_path, f"Unexpected result path. Expected {expected_path}, got {result}")

    # Test methods
    def test_long_sequence_to_class(self):
        self._run_embedding_test("long_sequences", 2, Protocol.sequence_to_class, include_short=False)

    def test_short_sequence_to_class(self):
        self._run_embedding_test("short_sequences", self.num_reads, Protocol.sequence_to_class, include_long=False)

    def test_mixed_sequence_to_class(self):
        self._run_embedding_test("mixed_sequences", self.num_reads, Protocol.sequence_to_class)

    def test_long_residue_to_class(self):
        self._run_embedding_test("long_sequences", 2, Protocol.residue_to_class, include_short=False)

    def test_short_residue_to_class(self):
        self._run_embedding_test("short_sequences", self.num_reads, Protocol.residue_to_class, include_long=False)

    def test_mixed_residue_to_class(self):
        self._run_embedding_test("mixed_sequences", self.num_reads, Protocol.residue_to_class)

    def test_not_allowed_seq_ids(self):
        seq_records = [BiotrainerSequenceRecord(seq_id="not/Allowed", seq="MMAAAG"),
                       BiotrainerSequenceRecord(seq_id="ALLOWED", seq="MMAAGX")
                       ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(ValueError):
                self.embedding_service.compute_embeddings(input_data=seq_records,
                                                          output_dir=Path(tmp_dir),
                                                          protocol=Protocol.sequence_to_class,
                                                          store_by_hash=False)


if __name__ == '__main__':
    unittest.main()
