# HF Dataset Integration

This configuration enables the use of datasets hosted on the Hugging Face repository. By specifying the `hf_dataset` option, **there is no need to have `sequence_file` and `labels_file` in your local machine**. Instead:

- The `sequence_file` and `labels_file` will be treated as the output file names for the data downloaded from the Hugging Face dataset and then processed.

## Protocol-Specific File Requirements

- For **per-sequence protocols** (`residues_to_class`, `residues_to_value`, `sequence_to_class`, `sequence_to_value`): Only the `sequence_file` option is needed.
- For **per-residue protocols** (`residue_to_class`): Both `sequence_file` and `targets_file` options are required.

---

## HF Dataset Configuration Options

The `hf_dataset` section of the configuration includes the following options:

- **`path` (required):** The repository path to the desired dataset in the Hugging Face hub (e.g., `huggingface_user_name/repository_name`).
- **`subset` (optional):** Specifies the subset of the dataset to download. 
  - If no subsets exist, you should remove this option or set it to `default`.
  - If the subset name is incorrect, an error will display the available subsets.
- **`sequence_column` (required):** The column in the dataset that contains the sequences.
- **`target_column` (required):** The column in the dataset that contains the targets.

---

## Handling Dataset Splits

Datasets in the Hugging Face repository may include predefined splits (e.g., `train`, `validation`, `test`). The tool handles splits as follows:

1. **If three predefined splits exist** (e.g., `train`, `validation`, `test`):
   - The splits are directly used as **TRAIN/VAL/TEST**.
2. **Otherwise**:
   - The entire dataset is merged, shuffled, and split into **TRAIN/VAL/TEST** with proportions of **70/15/15**.

This ensures compatibility with any dataset structure.

---

## Example Configuration

Below is an example YAML configuration using the `hf_dataset` option:

```yaml
sequence_file: sequences.fasta
labels_file: labels.fasta
protocol: residue_to_class
hf_dataset:
  path: heispv/protein_data_test
  subset: split_3
  sequence_column: protein_sequence
  target_column: secondary_structure
model_choice: FNN
optimizer_choice: adam
loss_choice: cross_entropy_loss
num_epochs: 200
use_class_weights: False
learning_rate: 1e-3
batch_size: 128
device: cpu
embedder_name: one_hot_encoding
```
or just run the code below:
```shell
poetry run biotrainer examples/hf_dataset/config.yml
```

**Notes**
- When using the `hf_dataset` option, the `sequence_file` and `labels_file` act as output file names for the downloaded and processed dataset.
- ure the `sequence_column` and `target_column` names match the structure of the dataset in the Hugging Face repository.

By following this configuration, you can seamlessly integrate Hugging Face datasets into your tool without requiring local sequence and label files. This setup also ensures proper handling of dataset splits for robust training, validation, and testing workflows.
