import os
import unittest
import tempfile

from pathlib import Path
from biotrainer.autoeval import AutoEval
from biotrainer.autoeval.pipelines.autoeval_setup import _apply_task_filter
from biotrainer_core.data_classes import ZeroShotMethod, BootstrappedMetric
from biotrainer_core.data_classes.autoeval import (
    AutoEvalTask, SupervisedFrameworkReport, UnsupervisedFrameworkReport,
    DEV_MODE_INDICATOR, DEV_MODE_ABLATED_INDICATOR
)
from biotrainer.autoeval.pipelines.autoeval_unsupervised import autoeval_unsupervised_pipeline
from biotrainer.autoeval.frameworks import PBCUnsupervisedFramework
from biotrainer.bioengineer import BioEngineer, BioEngineerBaseline


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
        """ Checks that autoeval pipeline runs correctly with zero-shot contact baseline in dev mode """
        TEST_CONTACT_STORAGE = Path(__file__).parent / "test_input_files"

        with tempfile.TemporaryDirectory() as tmp_dir_name:
            print("Starting AutoEval pipeline...")

            bio_engineer = BioEngineer.from_baseline(baseline=BioEngineerBaseline.RANDOM_BASELINE)
            autoeval = AutoEval(embedder_name="bioengineer_random_baseline",
                                output_dir=tmp_dir_name,
                                custom_bioengineer=bio_engineer,
                                custom_storage_path=TEST_CONTACT_STORAGE,
                                development_mode=True)
            report = autoeval.pbc_zeroshot_contact(zero_shot_method=ZeroShotMethod.JACOBIAN_CONTACT).run()

            self.assertTrue(report is not None)
            self.assertTrue(len(report.zeroshot_contact_results) > 0)
            contact_report = report.zeroshot_contact_results["PBC_ZEROSHOT_CONTACT"]
            self.assertIn(f"PBC_ZEROSHOT_CONTACT-test_dataset{DEV_MODE_INDICATOR}", contact_report.task_results)
            self.assertTrue(contact_report.used_development_mode())

    def test_autoeval_zeroshot_contact_full_eval_ablation(self):
        """ Checks that full eval computes both full task results and ablated task results """
        TEST_CONTACT_STORAGE = Path(__file__).parent / "test_input_files"

        with tempfile.TemporaryDirectory() as tmp_dir_name:
            print("Starting AutoEval pipeline...")

            bio_engineer = BioEngineer.from_baseline(baseline=BioEngineerBaseline.RANDOM_BASELINE)
            autoeval = AutoEval(embedder_name="bioengineer_random_baseline",
                                output_dir=tmp_dir_name,
                                custom_bioengineer=bio_engineer,
                                custom_storage_path=TEST_CONTACT_STORAGE,
                                development_mode=False)
            report = autoeval.pbc_zeroshot_contact(zero_shot_method=ZeroShotMethod.JACOBIAN_CONTACT).run()

            self.assertTrue(report is not None)
            self.assertTrue(len(report.zeroshot_contact_results) > 0)
            contact_report = report.zeroshot_contact_results["PBC_ZEROSHOT_CONTACT"]
            self.assertIn("PBC_ZEROSHOT_CONTACT-test_dataset", contact_report.task_results)
            self.assertIn(f"PBC_ZEROSHOT_CONTACT-test_dataset{DEV_MODE_ABLATED_INDICATOR}", contact_report.task_results)
            self.assertFalse(contact_report.used_development_mode())
            self.assertFalse(report.is_development())

    def test_autoeval_zeroshot_pipeline_ablation(self):
        """ Checks that autoeval_zeroshot_pipeline adds ablated results for DEV_DATASET_TEST frameworks """
        import shutil
        from biotrainer.autoeval.frameworks.pgym.pgym_framework import PGYMFramework
        from biotrainer.autoeval.pipelines.autoeval_zeroshot import autoeval_zeroshot_pipeline

        dataset_path = Path(__file__).parent / "test_input_files" / "pgym" / "B2L11_HUMAN_Dutta_2010_binding-Mcl-1.csv"

        with tempfile.TemporaryDirectory() as tmp_dir:
            file1 = Path(tmp_dir) / "file1.csv"
            file2 = Path(tmp_dir) / "file2.csv"
            shutil.copyfile(dataset_path, file1)
            shutil.copyfile(dataset_path, file2)

            task_full = AutoEvalTask(framework_name="PGYM", dataset_name="virus",
                                     input_files=[file1, file2], type="Protein")
            task_dev = AutoEvalTask(framework_name="PGYM", dataset_name=f"virus{DEV_MODE_INDICATOR}",
                                    input_files=[file1], type="Protein")

            bio_engineer = BioEngineer.from_baseline(baseline=BioEngineerBaseline.RANDOM_BASELINE)
            generator = autoeval_zeroshot_pipeline(
                framework=PGYMFramework(),
                embedder_name="bioengineer_random_baseline",
                zero_shot_method=ZeroShotMethod.WT_MARGINALS,
                output_dir=Path(tmp_dir),
                autoeval_tasks=[(task_full, {}), (task_dev, {})],
                development_mode=False,
                bioengineer=bio_engineer,
            )
            final_progress = None
            for progress in generator:
                final_progress = progress

            self.assertIsNotNone(final_progress)
            self.assertIsNotNone(final_progress.final_report)
            report = final_progress.final_report
            self.assertIn("PGYM-virus", report.task_results)
            self.assertIn(f"PGYM-virus{DEV_MODE_ABLATED_INDICATOR}", report.task_results)
            self.assertFalse(report.used_development_mode())

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

    def test_autoeval_dev_then_full_eval_preserves_framework_reports(self):
        """ Checks that running dev mode followed by full eval mode preserves all results in the framework report """
        TEST_CONTACT_STORAGE = Path(__file__).parent / "test_input_files"

        with tempfile.TemporaryDirectory() as tmp_dir_name:
            bio_engineer = BioEngineer.from_baseline(baseline=BioEngineerBaseline.RANDOM_BASELINE)

            # 1. Run in development mode
            autoeval_dev = AutoEval(embedder_name="bioengineer_random_baseline",
                                    output_dir=tmp_dir_name,
                                    custom_bioengineer=bio_engineer,
                                    custom_storage_path=TEST_CONTACT_STORAGE,
                                    development_mode=True)
            report_dev = autoeval_dev.pbc_zeroshot_contact(zero_shot_method=ZeroShotMethod.JACOBIAN_CONTACT).run()

            self.assertIn(f"PBC_ZEROSHOT_CONTACT-test_dataset{DEV_MODE_INDICATOR}",
                          report_dev.zeroshot_contact_results["PBC_ZEROSHOT_CONTACT"].task_results)

            # 2. Run in full evaluation mode with the same output directory
            autoeval_full = AutoEval(embedder_name="bioengineer_random_baseline",
                                     output_dir=tmp_dir_name,
                                     custom_bioengineer=bio_engineer,
                                     custom_storage_path=TEST_CONTACT_STORAGE,
                                     development_mode=False)
            report_full = autoeval_full.pbc_zeroshot_contact(zero_shot_method=ZeroShotMethod.JACOBIAN_CONTACT).run()

            contact_report = report_full.zeroshot_contact_results["PBC_ZEROSHOT_CONTACT"]
            # Both dev and full results must be present!
            self.assertIn(f"PBC_ZEROSHOT_CONTACT-test_dataset{DEV_MODE_INDICATOR}", contact_report.task_results)
            self.assertIn("PBC_ZEROSHOT_CONTACT-test_dataset", contact_report.task_results)
            self.assertIn(f"PBC_ZEROSHOT_CONTACT-test_dataset{DEV_MODE_ABLATED_INDICATOR}", contact_report.task_results)

    def test_zeroshot_pipeline_preserves_existing_report(self):
        """ Checks that autoeval_zeroshot_pipeline updates an existing report without deleting previous tasks """
        import shutil
        from biotrainer.autoeval.frameworks.pgym.pgym_framework import PGYMFramework
        from biotrainer.autoeval.pipelines.autoeval_zeroshot import autoeval_zeroshot_pipeline
        from biotrainer_core.data_classes.autoeval import ZeroShotFrameworkReport

        dataset_path = Path(__file__).parent / "test_input_files" / "pgym" / "B2L11_HUMAN_Dutta_2010_binding-Mcl-1.csv"

        with tempfile.TemporaryDirectory() as tmp_dir:
            file1 = Path(tmp_dir) / "file1.csv"
            shutil.copyfile(dataset_path, file1)

            task_dev = AutoEvalTask(framework_name="PGYM", dataset_name=f"virus{DEV_MODE_INDICATOR}",
                                    input_files=[file1], type="Protein")
            task_full = AutoEvalTask(framework_name="PGYM", dataset_name="virus",
                                     input_files=[file1], type="Protein")

            bio_engineer = BioEngineer.from_baseline(baseline=BioEngineerBaseline.RANDOM_BASELINE)
            # 1. Run dev task
            existing_report = ZeroShotFrameworkReport.empty(method=ZeroShotMethod.WT_MARGINALS)
            generator_dev = autoeval_zeroshot_pipeline(
                framework=PGYMFramework(),
                embedder_name="bioengineer_random_baseline",
                zero_shot_method=ZeroShotMethod.WT_MARGINALS,
                output_dir=Path(tmp_dir),
                autoeval_tasks=[(task_dev, {})],
                development_mode=True,
                framework_report=existing_report,
                bioengineer=bio_engineer,
            )
            for progress in generator_dev:
                pass
            self.assertIn(f"PGYM-virus{DEV_MODE_INDICATOR}", existing_report.task_results)

            # 2. Run full task with the same report handed over
            generator_full = autoeval_zeroshot_pipeline(
                framework=PGYMFramework(),
                embedder_name="bioengineer_random_baseline",
                zero_shot_method=ZeroShotMethod.WT_MARGINALS,
                output_dir=Path(tmp_dir),
                autoeval_tasks=[(task_full, {})],
                development_mode=False,
                framework_report=existing_report,
                bioengineer=bio_engineer,
            )
            for progress in generator_full:
                pass

            # Both dev and full tasks must be preserved in the report
            self.assertIn(f"PGYM-virus{DEV_MODE_INDICATOR}", existing_report.task_results)
            self.assertIn("PGYM-virus", existing_report.task_results)

    def test_unsupervised_pipeline_preserves_existing_report(self):
        """ Checks that autoeval_unsupervised_pipeline updates an existing report without deleting previous tasks """
        framework = PBCUnsupervisedFramework()
        existing_report = UnsupervisedFrameworkReport.empty(min_seq_len=0, max_seq_len=1000)
        existing_metric = BootstrappedMetric(name="accuracy", mean=0.8, lower=0.75, upper=0.85,
                                             iterations=1000, sample_size=10, confidence_level=0.05)
        existing_report.update_result("PBC_UNSUPERVISED-old_task (dev)", {"test": [existing_metric]})

        # When passed to pipeline or update_result, the previous task remains
        new_metric = BootstrappedMetric(name="accuracy", mean=0.9, lower=0.85, upper=0.95,
                                        iterations=1000, sample_size=20, confidence_level=0.05)
        existing_report.update_result("PBC_UNSUPERVISED-new_task", {"test": [new_metric]})

        self.assertIn("PBC_UNSUPERVISED-old_task (dev)", existing_report.task_results)
        self.assertIn("PBC_UNSUPERVISED-new_task", existing_report.task_results)

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
