# Biotrainer Changelog

## 28.06.2023 - Version 0.6.0
### Features
* Adding **bootstrapping** as a method to the Inferencer class. It allows to easily calculate error
margins for each metric. It can be called like this:
```python
result_dict = inferencer.from_embeddings_with_bootstrapping(per_residue_embeddings,
                                                            targets_r2c,
                                                            iterations=30,
                                                            seed=42)
```

### Maintenance
* Simplifying and re-using code for monte_carlo_dropout predictions for solvers

### Bug fixes
* Fixed monte carlo dropout predictions for per-residue protocols
* Fixed version in `version.py`

### Tests
* Adding tests for Inferencer module. At the moment, all inferencer API methods are covered for protocols
`residue_to_class` and `sequence_to_value`

## 26.06.2023 - Version 0.5.1
### Bug fixes
* Fixing bug that using a custom embedder script failed to create the log directory properly. This includes
moving the prohibited download check of `embedder_name` to the verify_config function of config.py.

## 30.05.2023 - Version 0.5.0
### Maintenance
* Adding a check in the `TargetManager.py` class that all provided splits are not empty.
This avoids getting an error after costly training if the test set was empty.
* Adding double-check if the cuda device from the out.yml file is available for the Inferencer 
module in `cuda_device.py`
* Simplifying the predict example.
Manual path correction is no longer necessary. Also added fix for mapped_predictions to show up correctly

## 18.04.2023 - Version 0.4.0
### Features
* Adding `CustomEmbedder`: It is now possible to use language models (embedders) that are not included in bio_embeddings
directly in *biotrainer*. See `examples/custom_embedder` for more information and hands-on instructions.
**This might introduce a security risk when running biotrainer as a remote service. Downloading of any custom_embedder
source file during execution is therefore disabled.** 

### Maintenance
* Updating dependencies. Enabled setup for the `torch.compile()` function of PyTorch 2.0. It is disabled for now
because it does not seem to be fully compatible with all our setups and models yet.
* Updating Dockerfile. Does now no longer include `bio_embeddings` by default. The docker example was adjusted.
* Adding `adam` as default `optimizer_choice` in `config.py`.

### Bug fixes
* Fixed logging and creation point of `log_dir` in `executer.py`

## 03.04.2023 - Version 0.3.1
### Bug fixes
* Fixing that using class weights for `residue_to_class` protocols did not work when providing a mask file.
**Class weights are now only calculated for the training dataset and for resolved residues (residue_to_x)!**
* Correcting citation

## 04.03.2023 - Version 0.3.0
### Features
* **Interaction mode**: Embeddings from two proteins can now be either multiplied (element-wise) or concatenated 
for protein-protein interaction prediction. This mode is not compatible with all protocols yet, 
tested throughout for `sequence_to_class`
* **Cross Validation**: Implemented *k_fold* and *leave_p_out* cross validation modes. The standard *hold_out* cross
validation with train/val/test sets is still the default. Splitting itself is done in the `cv_splitter.py` file of 
the trainer module. `auto_resume` also works with all versions of cross validation. If results are missing from a 
previous interrupted run, they are calculated again via inference by existing checkpoints. In addition, the metric
to choose the best model from the splits can be set manually (default: `choose_by: loss`)
* **Validation baselines**: `sanity_checker.py` is now able to calculate "zero-only", "one-only" baselines for 
binary classification tasks. Also adding a "mean-only" baseline for regression tasks.
The sanity checks can be disabled by a new flag in the config file: `sanity_check: False`.
Also computes the dataset bias and a "bias" baseline for interactions.
* **Monte-carlo-dropout inference**: `Inferencer.py` now supports monte-carlo dropout inference for models with 
dropout. This enables uncertainty quantification within the model for predictions
* Adding cli flag `--list-embedders` to show currently available embedders from `bio_embeddings`
* Adding logging to file in addition to logging to console (destination: `output_dir/logger_out.log`)
* Adding examples for *working with biotrainer files* and *protein_protein_interaction mode* 

### Maintenance
* Major refactorings for cross validation modes in the trainer module. Specifically, `trainer.py` now contains 
a `Trainer` class that handles the cross validation
* Moving `sanity_checker.py` to new module `validations` to prevent circular imports
* Moving `get_split_lists` from `target_manager_utils.py` to `FASTA.py` to have all fasta-related files in one place
* Adding a `__version__` tag to biotrainer module and to `out.yml` to keep track of the employed biotrainer version
for each run
* Set annotations can now be given via a simplified version, replacing the overcomplicated previous set annotations
  (`#New: SET=val #Old: SET=train VALIDATION=True`). The old version is still possible, enabling backwards compatibility
* Renaming `save_test_predictions` to `save_split_ids`. Sequence ids of all test/val splits can now be saved in 
combination with the test set predictions in order to reproduce the splits created by biotrainer
* Using torchmetrics for `SequenceRegressionSolver.py` instead of manually calculated *mean squared error*
* Removing `from_dict` function from Inferencer and moving its functionality to the `from_embeddings` function
* Adding a `create_from_out_file` method to Inferencer to simplify the creation of an 
Inferencer object from an out.yml file
* Adding random seed to Inferencer method `from_embeddings_with_monte_carlo_dropout` to keep predictions reproducible 

### Bug fixes
* **Fixing metrics calculation per epoch**: Previously, the mean over all batch results has been calculated, 
which is not correct for every metric or different batch sizes. 
**This change affects all classification tasks! Reported test results for classification tasks calculated with
prior versions are not reliable! 
However, if test_predictions have been stored, correct metrics still can be retrieved**

### Tests
* Adding tests for cross validation modes (`test_cross_validation.py`, hold_out, k_fold, leave_p_out)
* Adding tests for config files (`test_configurations.py`)
* Adding tests for bin_creation in cv_splitter (continuous values to bins for stratified k_fold cross validation,
`test_cv_splitter.py`)
* Adding tests for hp_search (random and grid search, checks if number of created hyperparameter combinations is 
correct, `test_hp_search.py`)

## 05.12.2022 - Version 0.2.1
### Bug fixes
* Fixing loss function not working on GPU (#62)
* Fixing incorrect metrics for classification task (#63)
* Fixing path to string for pretrained model (=> path is correctly saved in out.yml)

### Features
* Using device is now logged
* Adding a sanity_checker.py that checks if the test results have some obvious problems (like only predicting a single
class) (wip)
* Adding a `limited_sample_size` flag to train the model on a subset of all training ids. Makes it easy to check if the
model architecture is able to overfit on the training data
* Adding metrics from best training iteration to out.yml file (to compare with test set performance)
* Applying _validate_targets to all protocols in TargetManager

### Maintenance
* Conversion dataset -> torch.tensor moved to embeddings.py
* Storing training/validation/test ids is replaced with the amount of samples in the respective sets
* Storing start and end time in a reproducible, readable format
* Export of ConfigurationException via __init__.py file for consistency
* Removing unnecessary double-loading of checkpoint for test evaluation
* Adding typing to split lists in TargetManager

## 01.11.2022 - Initial Release: Version 0.2.0
### Features
* Protocols:
  * sequence to value
  * sequence to class
  * residues to class
  * residue to class
* Easy training on clusters
* Calculate embeddings via bio_embeddings
* Configuration tests
* Automatic check of input file consistency
* Standardization of input, output and embedding files


## 01.06.2022 - Prototype: Version 0.1.0