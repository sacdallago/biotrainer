import os
import unittest
import tempfile

from pathlib import Path
from biotrainer.autoeval import AutoEval
from biotrainer_core.data_classes import ZeroShotMethod
from biotrainer_core.data_classes.autoeval import ContactFrameworkReport, ZeroShotFrameworkReport
from biotrainer.bioengineer import BioEngineer, BioEngineerBaseline


class DevelopmentModeDetectionTests(unittest.TestCase):
    """ used_development_mode decides whether an existing report is re-run, so it must identify the results, not
    count them: an interrupted full run can hold exactly as many results as there are development ids. """

    def _contact_report(self, result_ids, development_ids):
        report = ContactFrameworkReport.empty(method=ZeroShotMethod.JACOBIAN_CONTACT)
        report.per_protein_results = {seq_id: None for seq_id in result_ids}
        report.development_ids = development_ids
        return report

    def _zero_shot_report(self, result_ids, development_ids):
        report = ZeroShotFrameworkReport.empty(method=ZeroShotMethod.MASKED_MARGINALS,
                                               development_ids=development_ids)
        report.individual_results = {dataset: None for dataset in result_ids}
        return report

    def test_development_run_is_detected(self):
        for report in [self._contact_report(["p1", "p2"], ["p1", "p2"]),
                       self._zero_shot_report(["d1.csv"], ["d1.csv"])]:
            self.assertTrue(report.used_development_mode())

    def test_full_run_is_not_detected_as_development(self):
        for report in [self._contact_report(["p1", "p2", "p3"], ["p1"]),
                       self._zero_shot_report(["d1.csv", "d2.csv"], ["d1.csv"])]:
            self.assertFalse(report.used_development_mode())

    def test_interrupted_full_run_is_not_detected_as_development(self):
        """ Same number of results as development ids, but not the same proteins """
        for report in [self._contact_report(["p1", "p9"], ["p1", "p2"]),
                       self._zero_shot_report(["d9.csv"], ["d1.csv"])]:
            self.assertFalse(report.used_development_mode())


class AutoevalTests(unittest.TestCase):

    @unittest.skipUnless(os.getenv('CI'), "Slow test - only run in CI")
    def test_autoeval_pbc_supervised_ohe(self):
        """ Checks that autoeval pipeline runs correctly with one hot encoding """
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            print("Starting AutoEval pipeline...")

            autoeval = AutoEval(embedder_name="one_hot_encoding",
                                output_dir=tmp_dir_name,
                                min_seq_length=10,
                                max_seq_length=450)
            report = autoeval.pbc_supervised().run()

            self.assertTrue(report is not None)
            self.assertTrue(len(report.supervised_results) > 0)

    def test_autoeval_zeroshot_contact_baseline(self):
        """ Checks that autoeval pipeline runs correctly with zero-shot contact baseline """
        TEST_CONTACT_STORAGE = Path(__file__).parent / "test_input_files"

        with tempfile.TemporaryDirectory() as tmp_dir_name:
            print("Starting AutoEval pipeline...")

            bio_engineer = BioEngineer.from_baseline(baseline=BioEngineerBaseline.RANDOM_BASELINE)
            autoeval = AutoEval(embedder_name="bioengineer_random_baseline",
                                output_dir=tmp_dir_name,
                                custom_bioengineer=bio_engineer,
                                custom_storage_path=TEST_CONTACT_STORAGE, )
            report = autoeval.pbc_zeroshot_contact(zero_shot_method=ZeroShotMethod.JACOBIAN_CONTACT).run()

            self.assertTrue(report is not None)
            self.assertTrue(len(report.zeroshot_contact_results) > 0)


    @unittest.skip(reason="Large test that should only be executed on demand")
    def test_autoeval_zeroshot_contact_ESM2_8M_UR50D(self):
        """ Checks that autoeval pipeline runs correctly with zero-shot contact ESM2_8M_UR50D """
        TEST_CONTACT_STORAGE = Path(__file__).parent / "test_input_files"

        with tempfile.TemporaryDirectory() as tmp_dir_name:
            print("Starting AutoEval pipeline...")

            autoeval = AutoEval(embedder_name="facebook/esm2_t6_8M_UR50D",
                                output_dir=tmp_dir_name,
                                custom_storage_path=TEST_CONTACT_STORAGE)
            report = autoeval.pbc_zeroshot_contact(zero_shot_method=ZeroShotMethod.JACOBIAN_CONTACT).run()

            self.assertTrue(report is not None)
            self.assertTrue(len(report.zeroshot_contact_results) > 0)
