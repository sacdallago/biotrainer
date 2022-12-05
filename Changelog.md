# Biotrainer Changelog

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