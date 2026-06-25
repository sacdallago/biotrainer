import os
import unittest
import tempfile
from pathlib import Path

from biotrainer.autoeval import autoeval_pipeline
from biotrainer.bioengineer import BioEngineer, BioEngineerBaseline, ZeroShotMethod


class AutoevalTests(unittest.TestCase):

    @unittest.skipUnless(os.getenv('CI'), "Slow test - only run in CI")
    def test_autoeval_ohe(self):
        """ Checks that autoeval pipeline runs correctly with one hot encoding """
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            print("Starting AutoEval pipeline...")

            current_progress = None
            for progress in autoeval_pipeline(embedder_name="one_hot_encoding",
                                              framework="PBC",
                                              output_dir=tmp_dir_name,
                                              min_seq_length=10,
                                              max_seq_length=450,
                                              ):
                print(progress)
                self.assertTrue(progress.current_framework_name == "PBC")
                current_progress = progress

            self.assertIsNotNone(current_progress)
            self.assertTrue(current_progress.final_report is not None)
            self.assertTrue(current_progress.completed_tasks == current_progress.total_tasks)
            self.assertTrue(len(current_progress.final_report.supervised_results) > 0)
            self.assertTrue(len(current_progress.final_report.supervised_results["PBC"].results)
                            == current_progress.total_tasks)


    def test_autoeval_zeroshot_contact_baseline(self):
        """ Checks that autoeval pipeline runs correctly with zero-shot contact baseline """
        TEST_CONTACT_STORAGE = Path(__file__).parent / "test_input_files"

        with tempfile.TemporaryDirectory() as tmp_dir_name:
            print("Starting AutoEval pipeline...")

            current_progress = None
            bio_engineer = BioEngineer.from_baseline(baseline=BioEngineerBaseline.RANDOM_BASELINE)
            for progress in autoeval_pipeline(embedder_name="bioengineer_random_baseline",
                                            framework="ZEROSHOT_CONTACT",
                                            zero_shot_method=ZeroShotMethod.JACOBIAN_CONTACT,
                                            output_dir=tmp_dir_name,
                                            custom_bioengineer=bio_engineer,
                                            custom_storage_path=TEST_CONTACT_STORAGE,
                                            ):
                print(progress)
                self.assertTrue(progress.current_framework_name == "ZEROSHOT_CONTACT")
                current_progress = progress

            self.assertIsNotNone(current_progress)
            self.assertTrue(current_progress.final_report is not None)
            self.assertTrue(current_progress.completed_tasks == current_progress.total_tasks)
            # TODO: add additional assertions / tests!


    @unittest.skip(reason="Large test that should only be executed on demand")
    def test_autoeval_zeroshot_contact_ESM2_8M_UR50D(self):
        """ Checks that autoeval pipeline runs correctly with zero-shot contact ESM2_8M_UR50D """
        TEST_CONTACT_STORAGE = Path(__file__).parent / "test_input_files"

        with tempfile.TemporaryDirectory() as tmp_dir_name:
            print("Starting AutoEval pipeline...")

            current_progress = None
            for progress in autoeval_pipeline(embedder_name="facebook/esm2_t6_8M_UR50D",
                                            framework="ZEROSHOT_CONTACT",
                                            zero_shot_method=ZeroShotMethod.JACOBIAN_CONTACT,
                                            output_dir=tmp_dir_name,
                                            custom_storage_path=TEST_CONTACT_STORAGE,
                                            ):
                print(progress)
                self.assertTrue(progress.current_framework_name == "ZEROSHOT_CONTACT")
                current_progress = progress

            self.assertIsNotNone(current_progress)
            self.assertTrue(current_progress.final_report is not None)
            self.assertTrue(current_progress.completed_tasks == current_progress.total_tasks)
            # TODO: add additional assertions / tests!