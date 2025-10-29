import unittest
import tempfile

from biotrainer.autoeval import autoeval_pipeline


class AutoevalTests(unittest.TestCase):

    def test_autoeval_ohe(self):
        """ Checks that autoeval pipeline runs correctly with one hot encoding """
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            print("Starting AutoEval pipeline...")

            current_progress = None
            for progress in autoeval_pipeline(embedder_name="one_hot_encoding",
                                              framework="pbc",
                                              output_dir=tmp_dir_name,
                                              min_seq_length=300,
                                              max_seq_length=350,
                                              ):
                print(progress)
                self.assertTrue(progress.current_framework_name == "pbc")
                current_progress = progress

            self.assertIsNotNone(current_progress)
            self.assertTrue(current_progress.final_report is not None)
            self.assertTrue(len(current_progress.final_report['results']) == current_progress.completed_tasks)
