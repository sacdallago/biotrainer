# Biotrainer Changelog

## 18.11.2025 - Version 1.2.1

## Fixes

- Added additional verifications for h5 files and embeddings calculation to improve reliability in `autoeval_pipeline`.
- Fixed issues with example embedding functions in the `plm_eval` Jupyter notebook for accurate demonstrations.
- Using Pydantic for `BiotrainerSeqRecord` class for validation and consistency.
- Exported `OutputData` in the `output_files` module for better modularity.

## Features

- Introduced a new `input_data` configuration option, allowing sequence records to be directly provided in code for
  flexibility.

## 29.10.2025 - Version 1.2.0

## Feature

- Added autoeval framework PBC and example notebook.
- Introduced autoeval_pipeline parameters for loading precomputed embeddings.
- Added automatic dimensionality reduction to the inferencer.

## Fixes

- Replaced t-SNE with PCA for embeddings reduction (because t-SNE cannot be retroactively applied).
- Fixed embedding functions compatibility with class_weights and classification protocols.
- Updated dataset links to point to the PBC repository.
- Corrected issues in autoeval tutorial notebooks and inference test configurations.

## Refactors

- Added support for Python 3.13 [BETA].
- Enhanced flexibility for test set naming and output paths.
- Centralized shared functionality of data handler in the base class.

## Tests

- Added unit tests for autoeval pipeline and class weight training validation.

## 25.09.2025 - Version 1.1.0

## Feature

* Adding **blosum62** predefined embedder via the `blosum` python package using the blosum substitution matrix as
  embeddings
* Adding **AAOntology** predefined embedder from https://doi.org/10.1016/j.jmb.2024.168717 using amino acid feature
  scales
* Adding biotrainer-ready quickstart
  datasets ([subcellular location](https://github.com/J-SNACKKB/FLIP/tree/main/splits/scl)
  and [secondary structure](https://github.com/J-SNACKKB/FLIP/tree/main/splits/secondary_structure)) in the `README.md`
* Adding masked language modeling (MLM) task via residue_to_class protocol, CNN decoder and `random_masking` option in
  finetuning config
* Adding lora examples for MLM and downstream tasks
* [BETA] Adding `residue_to_value` protocol

## Breaking

* Refactoring confidence range calculation to use empirical distribution.
  Bootstrapping and MCD used assumption of normal distribution, which is okay for large sample sizes due to CLT. But it
  is more appropriate to use the empirical distribution, giving better upper and lower bounds especially for small
  sample sizes
* autoeval: Adding framework name to task name in autoeval. This makes it easier to add multiple frameworks in the
  future
* autoeval: Changing autoeval FLIP scl protocol to sequence_to_class. This requires less resources but is also valid
  to evaluate plms

## Maintenance

* Updating dependencies

## Fixes

* Fixing broken use_half_precision embeddings mode and adding comment about downstream float32 precision usage

## 03.07.2025 - Version 1.0.0

## Feature

* Adding an `OutputManager` class that can be customized by adding observers for easier integration with external
  tools such as MLFlow, WanB and tensorboard (the latter is already supported)
* Adding the `autoeval` module to biotrainer that enables evaluating protein language models on downstream tasks.
  Currently, a curated subset of the [FLIP](https://github.com/J-SNACKKB/FLIP) datasets is supported.
* Adding an improved CLI, including `train`, `predict`, `convert` (deprecated files) and `autoeval` commands
* Adding an `InputValidator` and `InputValidationStep` that validates the given `input_file`. Can be
  deactivated by setting `validate_input` to `False` in the config file.
* Adding **LoRA** finetuning via `finetuning_config`. Implementation is currently in beta state
  (some modes like auto_resume and ppi are not supported), but finetuning can already be applied for all protocols.
* Adding a `random_embedder` to calculate random embeddings as a baseline for predefined embedders.

## Maintenance

* Replaced biopython dependency with custom read, write and filter functions
* Refactored the large trainer class into a pipeline with distinct steps for better readability, maintainability and
  customization
* Enforced bootstrapping for sanity checks
* Refactoring embedding_service to allow embedding computation as generator function. Embeddings are now directly
  stored in the h5 file after computation. Experiments show that this is about as efficient as the old batching
  approach,
  while allowing for better code readability.
* Adding PyPi release
* Adding official macOS support

## Breaking

* Refactoring file input to a single `input_file`.
  `sequence_file`, `labels_file` and `mask_file` are no longer supported.
* Naming changes in the output file, documented in this issue: https://github.com/sacdallago/biotrainer/issues/137
* `embedder_name` and `embeddings_file` are no longer mutually exclusive. If an `embeddings_file` is provided,
  it will be used instead of calculating the embeddings
* Embeddings are now stored by hash in the result h5 file. The behaviour can be turned off for special use-cases
  in the `compute_embeddings` function by setting the `store_by_hash` flag to `False`. In that case, the original
  sequence id is now used (over a running integer) as h5 index. The sequence id also always saved in the `original_id`
  attribute of the h5 dataset.
* Ending support for Python 3.10, adding support for Python 3.12
* Migrating build and dependency system from `poetry` to `uv` for better performance

## 05.05.2025 - Version 0.9.8

### Features

* Adding multiple distinct test sets (e.g. >Seq SET=test ... >Seq2 SET=test2 ... >Seq3 SET=test5 ...)
* Adding a prediction set (unlabeled sequence,
  that is predicted with the best model after the training, e.g. >SeqPred SET=pred)
* Adding a unique model hash (16 chars sha256, based on input files, config options and custom trainer)
* Improving monte carlo dropout predictions: Adding new test cases, checks for correct parameters and
  `all_predictions` and `confidence_range` to result dict

### Maintenance

* Adding validation for per-residue masks (must each contain at least one resolved (1) value)
* Adapting .pt API usage for new PyTorch version
* Updating dependencies

## 08.04.2025 - Version 0.9.7

### Features

* Adding `external_writer` config option to (de-)activate external writer
* Adding embedding via onnx embedder. A `custom_tokenizer_config` can be provided if necessary (default: T5Tokenizer).
  For further information, see the "exmaples/onnx_embedder" directory.
* Adding heuristic to set default batch size for embedder models
* Adding new cli function `headless_main_with_custom_trainer`. It can be used to provide a custom trainer that allows
  to overwrite most functionality of biotrainer. Note that using this will add the entry "custom_trainer: True" to the
  `out.yml` file.
* Improving biotrainer training output in jupyter notebook environments

## 06.02.2025 - Version 0.9.6

### Features

* Adding a random model baseline (model with randomly initialized weights)
* Adding bootstrapping to all sanity check baselines
* Adding a biotrainer intro notebook about creating a model with
  one_hot_encodings and compare the results to an existing model

### Maintenance

* Disabling the test of very long sequences in test_embeddings_service.py for the CI.
  This should speed up the CI considerably.
* Simplifying the config module by removing config options as classes and replacing them with dataclasses-based objects
* Adding a new [overview](docs/config_file_options_overview.md) about all available config options
* Improve stability of ONNX conversion, especially on Windows
* Making bootstrapping results reproducible with a seeded number generator
* Improve the [first_steps.md](docs/first_steps.md) tutorial
* Simplifying readme
* Update dependencies

## 09.12.2024 - Version 0.9.5

### Features

* Added integration for huggingface datasets by @heispv in https://github.com/sacdallago/biotrainer/pull/124
* Added per-sequence dimension reduction methods by @nadyadevani3112
  in https://github.com/sacdallago/biotrainer/pull/123
* Improving one_hot_encoding embedder with numpy functions

### Maintenance

* Fixing "precis<s>s</s>ion" typo in `clasification_solver.py`
* Updating dependencies
* Improving documentation of the config module by @heispv in https://github.com/sacdallago/biotrainer/pull/121
* Improving `compute_embeddings` function to handle `Dict`, `str` and `Path` as `input_data`
* Reducing log level of onnx and dynamo to ERROR to decrease logging output
* Fixing first_steps documentation
* Adding links to biocentral app, repository and biotrainer documentation

## 29.10.2024 - Version 0.9.4

### Bug fixes

* Hotfix for incorrect precision mode setting by @SebieF in https://github.com/sacdallago/biotrainer/pull/116

### Maintenance

* Updating dependencies: **removing python3.9 support**
* Updating CI workflow to be compatible with Windows

## 14.10.2024 - Version 0.9.3

### Features

* Adding support for ProstT5 embedder by @SebieF in https://github.com/sacdallago/biotrainer/pull/110

### Maintenance

* Adding troubleshooting guide by @SebieF in https://github.com/sacdallago/biotrainer/pull/109

### Bug fixes

* Adding improved onnx saving and inferencer fixes by @SebieF in https://github.com/sacdallago/biotrainer/pull/112

## 26.08.2024 - Version 0.9.2

### Features

* Improving memory management of embedding calculation by @SebieF in https://github.com/sacdallago/biotrainer/pull/96
* Use a strategy for sequence preprocessing by @SebieF in https://github.com/sacdallago/biotrainer/pull/99
* Adding ONNX support by @SebieF in https://github.com/sacdallago/biotrainer/pull/101
* Adding saprot embedder example by @SebieF in https://github.com/sacdallago/biotrainer/pull/106

### Maintenance

* **BREAKING** Improving masking mechanisms in CNN and LightAttention models by @SebieF
  in https://github.com/sacdallago/biotrainer/pull/102
* Improving embedder model and tokenizer class recognition by @SebieF
  in https://github.com/sacdallago/biotrainer/pull/105
* Optimize Memory Handling in Embedding Computations and Refactor EmbeddingService by @heispv
  in https://github.com/sacdallago/biotrainer/pull/103
* Updating dependencies

## 10.07.2024 - Version 0.9.1

### Maintenance

* Fixing error in type checking for device
* Updating dependencies
* Updating inference examples
* Adding hint for version mismatch in inferencer
* Adding class weights to `out.yml` if they are calculated

### Features

* Improving fallback mechanism of embedder models. Now, cpu mode is exited once there is enough
  RAM again for shorter sequences
* Changing model storage format from `.pt` to `.safetensors`.
  Safetensors is safer for model sharing. Legacy `.pt` format is still supported, and can be converted via

```python
from biotrainer.inference import Inferencer

inferencer, out_file = Inferencer.create_from_out_file(out_file_path="out.yml", allow_torch_pt_loading=True)
inferencer.convert_all_checkpoints_to_safetensors()
```

## 09.06.2024 - Version 0.9.0

### Maintenance

* Adding more extensive code documentation
* Optimizing imports
* Applying consistent file naming
* Updating dependencies. Note that `jupyter` was removed as a direct optional dependency.
  You can always add it via `poetry add jupyter`.
* Adding simple differentiation between t5 and esm tokenizer and models in `embedders` module

### Features

* Adding new `residues_to_value` protocol.
  Similar to the residues_to_class protocol,
  this protocol predicts a value for each sequence, using per-residue embeddings. It might, in some situations,
  outperform
  the sequence_to_value protocol.

### Bug fixes

* For `huggingface_transformer_embedder.py`, all special tokens are now always deleted from the final embedding
  (e.g. first/last for esm1b, last for t5)

## 04.06.2024 - Version 0.8.4

### Maintenance

* Updating dependencies
* Adding pip-audit dependency check to CI pipeline

## 04.05.2024 - Version 0.8.3

### Maintenance

* Updating dependencies

### Features

* Adding mps device for macOS. Use by setting the following configuration option: `device: mps`.
  *Note that MPS is still under development, use it at your responsibility.*
* Adding flags to the `compute_embedding` method of `EmbeddingService`

1. `force_output_dir`: Do not change the given output directory within the method
2. `force_recomputing`: Always re-compute the embeddings, even if an existing file is found

These changes are made to make the embedders module of biotrainer easier usable outside the biotrainer pipeline itself.

## 27.02.2024 - Version 0.8.2

### Maintenance

* Updating dependencies

### Features

* Adding option to ignore verification of files in `configurator.py`. This makes it possible to verify a biotrainer
  configuration independently of the provided files.
* Adding new compute_embeddings_from_list function to embedding_service.py. This allows to compute embeddings directly
  from sequence strings.

## 12.01.2024 - Version 0.8.1

### Maintenance

* Updating dependencies after removing bio_embeddings, notably upgrading torch and adding accelerate
* Updating examples, documentation, config and test files for inferencer tests to match the new compile mode
* Replaced the exception with a warning if dropout_rate was set for a model that does not support it (e.g. LogReg)

### Features

* Enable pytorch compile mode. The feature exists since torch 2.0 and is now available in biotrainer. It can be
  enabled via

```yaml
disable_pytorch_compile: False
```

## 04.01.2024 - Version 0.8.0

### Maintenance

* Removing dependency on *bio_embeddings* entirely. *bio_embeddings* is not really maintained
  anymore (last commit 2 years ago) and being dependent on a specific external module for embeddings calculation
  shrinks the overall capabilities of biotrainer. Now, for example, adding LORA layers becomes much easier.
  While *bio_embeddings* does have its advantages such as a well-defined pipeline and a lot of utilities, it also
  provides a lot of functionalities that is not used by biotrainer. Therefore, a new `embedders` module was introduced
  to biotrainer that mimics some aspects of *bio_embeddings* and takes inspiration from it. However, it is built in a
  more
  generic way and enables, in principle, all huggingface transformer embedders to be used by biotrainer.
* Ankh custom embedder was removed, because it can now be used directly in biotrainer:

```yaml
embedder_name: ElnaggarLab/ankh-large
```

* Adding new use_half_precision option for transformer embedders
* Adding missing device option

### Bug fixes

* Fixed a minor problem for model saving in `Solver.py`:
  If a new model was trained, and it does not improve until `early_stop` is triggered, it was not saved as a checkpoint.

## 08.09.2023 - Version 0.7.0

### Maintenance

* Added `config` module to read and verify the given config file. It was decided to refactor the handling of the
  configuration file to allow for higher complexity of the config and to be able to expose the configuration options
  to third party applications, such as servers and file linters. This should pay off for the increase in code
  complexity.
  All config option classes are as much encapsulated as possible. They are able to validate their given value and
  transform it if necessary (e.g. making file paths absolute or downloading files).
  In addition to the option classes, rules have been defined which can be applied to the whole configuration file.
  They can, amongst others, be used to define mutual exclusive or required options and files,
  depending on the protocol of the value of other options.
* Updating dependencies

### Tests

* Added new unit tests to check the config module

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
* Changing confidence interval calculation for monte_carlo_dropout predictions and bootstrapping.
  The number of iterations is now no longer included for calculating the interval:

```python
std_dev, mean = torch.std_mean(values, dim=dimension, unbiased=True)
# Use normal distribution for critical value (z_score)
z_score = norm.ppf(q=1 - (confidence_level / 2))
# Confidence range does not include number of iterations:
# https://moderndive.com/8-confidence-intervals.html#se-method
# Note that the number of iterations influences the precision of the standard deviation, however.
confidence_range = z_score * std_dev
```

### Bug fixes

* Fixed monte carlo dropout predictions for per-residue protocols
* Fixed version in `version.py`

### Tests

* Adding tests for Inferencer module. All inferencer API methods are covered for all protocols

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