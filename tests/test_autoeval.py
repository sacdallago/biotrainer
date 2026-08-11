import os
import unittest
import tempfile

import numpy as np

from pathlib import Path
from types import SimpleNamespace
from biotrainer.autoeval import AutoEval
from biotrainer.autoeval.pipelines.autoeval_setup import _apply_task_filter
from biotrainer.autoeval.pipelines.autoeval_zeroshot_contact import autoeval_zeroshot_contact_pipeline
from biotrainer_core.data_classes import ZeroShotMethod
from biotrainer_core.data_classes.autoeval import AutoEvalTask, ContactFrameworkReport, \
    SupervisedFrameworkReport, ZeroShotFrameworkReport
from biotrainer.bioengineer import BioEngineer, BioEngineerBaseline
from biotrainer.shared import SequenceTooLongError


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


class ZeroShotContactSkipTests(unittest.TestCase):
    """ A skipped protein must not turn a development-mode report into one that looks like a full evaluation:
    used_development_mode() compares the results against the development ids as sets, so an id whose protein
    was skipped has to leave the development ids too - otherwise the partial report is served later instead of
    running the full evaluation. """

    dataset_dir = Path(__file__).parent / "test_input_files" / "CONTACT" / "contacts" / "zeroshot" / "test_dataset"

    def _run_development_pipeline(self, bioengineer, output_dir) -> ContactFrameworkReport:
        task = AutoEvalTask(framework_name="PBC_ZEROSHOT_CONTACT", dataset_name="test_dataset",
                            input_files=[self.dataset_dir], type="Protein")
        progress = list(autoeval_zeroshot_contact_pipeline(
            framework=SimpleNamespace(get_name=lambda: "PBC_ZEROSHOT_CONTACT"),
            embedder_name="bioengineer_random_baseline",
            zero_shot_method=ZeroShotMethod.JACOBIAN_CONTACT,
            output_dir=Path(output_dir),
            autoeval_tasks=[(task, {})],
            bioengineer=bioengineer,
            development_mode=True))
        return progress[-1].final_report

    def test_skipped_protein_keeps_the_report_marked_as_development(self):
        skipped_sequences = []

        class _SkipsTheFirstProtein:
            def zero_shot_contact_map(self, method, sequence, batch_size: int = 32):
                if not skipped_sequences:
                    skipped_sequences.append(sequence)
                    raise SequenceTooLongError("2000 tokens exceed the 1024 token context")
                return np.random.default_rng(0).random((len(sequence), len(sequence)))

        with tempfile.TemporaryDirectory() as tmp_dir_name:
            report = self._run_development_pipeline(_SkipsTheFirstProtein(), tmp_dir_name)

        self.assertEqual(len(skipped_sequences), 1, "Test setup: exactly one protein should have been skipped")
        self.assertEqual(len(report.development_ids), len(report.per_protein_results),
                         "The skipped protein is missing from the results, so it may not stay a development id!")
        self.assertTrue(report.used_development_mode(),
                        "A development run that skipped a protein would be served as a full evaluation!")


class TaskFilterTests(unittest.TestCase):
    """ task_filter restricts a framework to a subset of its tasks, and the subset has to stay visible in the
    report - otherwise a later full run finds a report and skips the framework. """

    @staticmethod
    def _task(dataset_name: str) -> AutoEvalTask:
        return AutoEvalTask(framework_name="PBC_SUPERVISED", dataset_name=dataset_name,
                            input_files=[Path(f"{dataset_name}.fasta")], type="Protein")

    def setUp(self):
        self.tasks = [self._task("scl"), self._task("disorder"), self._task("gb1")]

    def test_filter_selects_subset(self):
        selected = _apply_task_filter(tasks=self.tasks,
                                      task_filter=lambda task: task.dataset_name == "scl",
                                      framework_name="PBC_SUPERVISED")
        self.assertEqual([task.dataset_name for task in selected], ["scl"])

    def test_filter_matching_nothing_raises(self):
        with self.assertRaisesRegex(ValueError, "selected none of the 3 tasks") as context:
            _apply_task_filter(tasks=self.tasks,
                               task_filter=lambda task: task.dataset_name == "typo",
                               framework_name="PBC_SUPERVISED")
        # The error is only actionable if it names what could have been selected
        self.assertIn("PBC_SUPERVISED-scl", str(context.exception))

    def test_report_records_that_a_filter_was_applied(self):
        report = SupervisedFrameworkReport.empty(min_seq_len=None, max_seq_len=None)
        self.assertFalse(report.task_filter_applied,
                         "Reports written before task filtering existed must not read as filtered!")
        report.task_filter_applied = True
        # _general_task_setup skips a framework whose report is complete - a filtered one never is
        self.assertTrue(report.task_filter_applied)


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

    def test_autoeval_task_filter_matching_nothing_raises(self):
        """ A filter that selects no task must fail the run instead of writing an empty report """
        TEST_CONTACT_STORAGE = Path(__file__).parent / "test_input_files"

        with tempfile.TemporaryDirectory() as tmp_dir_name:
            autoeval = AutoEval(embedder_name="bioengineer_random_baseline",
                                output_dir=tmp_dir_name,
                                custom_bioengineer=BioEngineer.from_baseline(
                                    baseline=BioEngineerBaseline.RANDOM_BASELINE),
                                custom_storage_path=TEST_CONTACT_STORAGE, )
            with self.assertRaisesRegex(ValueError, "selected none of the"):
                autoeval.pbc_zeroshot_contact(
                    zero_shot_method=ZeroShotMethod.JACOBIAN_CONTACT,
                    task_filter=lambda task: task.dataset_name == "no_such_dataset").run()

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
