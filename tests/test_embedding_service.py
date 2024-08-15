import unittest
import logging
import tempfile
import os
import random
import psutil
import shutil
import time
from pathlib import Path

from biotrainer.embedders import OneHotEncodingEmbedder, EmbeddingService
from biotrainer.protocols import Protocol

logger = logging.getLogger(__name__)

class TestEmbeddingService(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._configure_logging()
    
    @classmethod
    def _configure_logging(cls):
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler()]
        )
        logger.info('Setting up the test case class')

    def setUp(self):
        self._setup_test_environment()
        self._setup_fasta_parameters()

    def _setup_test_environment(self):
        logger.info('Setting up test environment')
        self.embedder = OneHotEncodingEmbedder()
        self.embedding_service = EmbeddingService(embedder=self.embedder)
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = Path(self.temp_dir)

    def _setup_fasta_parameters(self):
        self.num_reads = 100
        self.long_length = int((0.75 * psutil.virtual_memory().available) / (18 * 21))
        self.other_length = 250

    def tearDown(self):
        self._cleanup_test_environment()

    def _cleanup_test_environment(self):
        logger.info('Tearing down test environment')
        max_retries = 5
        for attempt in range(max_retries):
            try:
                shutil.rmtree(self.temp_dir)
                break
            except Exception as e:
                logger.warning(f'Attempt {attempt + 1} failed to remove temporary directory {self.temp_dir}: {str(e)}')
                if attempt < max_retries - 1:
                    time.sleep(0.5)  # Wait for half a second before retrying
                else:
                    logger.error(f'Failed to remove temporary directory after {max_retries} attempts')
                    self._remove_files_individually()

    def _remove_files_individually(self):
        for root, dirs, files in os.walk(self.temp_dir, topdown=False):
            for name in files:
                try:
                    os.unlink(os.path.join(root, name))
                except Exception as e:
                    logger.error(f'Failed to remove file {name}: {str(e)}')
            for name in dirs:
                try:
                    os.rmdir(os.path.join(root, name))
                except Exception as e:
                    logger.error(f'Failed to remove directory {name}: {str(e)}')
        try:
            os.rmdir(self.temp_dir)
        except Exception as e:
            logger.error(f'Failed to remove root temporary directory {self.temp_dir}: {str(e)}')

    @staticmethod
    def _generate_sequence(length):
        """Generate a random sequence of a given length."""
        return ''.join(random.choice("ACDE") for _ in range(length))

    def _generate_fasta(self, num_reads, long_length, other_length, filename, include_long = True, include_short = True):
        """Generate a FASTA file with a specified number of reads."""
        logger.info(f"Generating FASTA file: {filename}")
        with open(filename, 'w') as file:
            if include_long:
                logger.info(f"Generating long sequence, it may take a bit of time.")
                for i in range(1, 3):
                    file.write(f">read_{i}\n{self._generate_sequence(long_length)}\n")
            
            if include_short:
                for i in range(3, num_reads + 1):
                    file.write(f">read_{i}\n{self._generate_sequence(other_length)}\n")
        logger.info(f"FASTA file generated: {filename}")

    def test_compute_embeddings_long_sequences(self):
        self._run_embedding_test("long_sequences", 2, include_short=False)

    def test_compute_embeddings_short_sequences(self):
        self._run_embedding_test("short_sequences", self.num_reads, include_long=False)

    def test_compute_embeddings_mixed_sequences(self):
        self._run_embedding_test("mixed_sequences", self.num_reads)

    def _run_embedding_test(self, test_name, num_reads, include_long=True, include_short=True):
        logger.info(f'Starting test: compute_embeddings_{test_name}')
        sequence_path = os.path.join(self.temp_dir, f"{test_name}.fasta")
        self._generate_test_fasta(sequence_path, num_reads, include_long, include_short)
        result = self._compute_embeddings(sequence_path)
        self._verify_result(result)
        logger.info(f'Test compute_embeddings_{test_name} completed successfully')

    def _generate_test_fasta(self, sequence_path, num_reads, include_long, include_short):
        logger.info(f"Generating FASTA file: {sequence_path}")
        self._generate_fasta(num_reads, self.long_length, self.other_length, sequence_path, include_long, include_short)

    def _compute_embeddings(self, sequence_path):
        logger.info('Computing embeddings')
        return self.embedding_service.compute_embeddings(
            sequence_path,
            self.output_dir,
            Protocol.sequence_to_class
        )

    def _verify_result(self, result):
        self.assertTrue(os.path.exists(result), f"Result file does not exist: {result}")
        expected_path = os.path.join(self.temp_dir, "sequence_to_class", "one_hot_encoding", "reduced_embeddings_file_one_hot_encoding.h5")
        self.assertEqual(result, expected_path, f"Unexpected result path. Expected {expected_path}, got {result}")

if __name__ == '__main__':
    unittest.main()