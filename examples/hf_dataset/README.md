# HF Dataset Integration

This integration allows you to use datasets hosted on HuggingFace without needing local `sequence_file`, `labels_file`, or `mask_file`. The tool creates these files automatically in the `hf_db` folder based on your config.

### Configuration

In your `config.yml`, set the following options under `hf_dataset`:

```yaml
hf_dataset:
  path: "huggingface_user_name/repository_name"
  subset: "subset_name_if_there_are"
  sequence_column: "column_for_sequences"
  target_column: "column_for_targets"
  mask_column: "optional_column_for_masks"
```

### Dataset Splits

If the dataset includes predefined splits like `train`, `validation`, or `test`, they will be used directly. Otherwise, a `ConfigurationException` will occur.

### Example

Run the example with:

```bash
poetry run biotrainer examples/hf_dataset/config.yml
```

**Notes**
- When using the `hf_dataset` option, remove the `sequence_file`, `labels_file`, and `mask_file` entries from the config.
- Ensure that the `sequence_column`, `target_column`, and `mask_column` names match the structure of the dataset in the HuggingFace repository.

By following this configuration, you can seamlessly integrate HuggingFace datasets into your tool without requiring local sequence and label files. This setup also ensures proper handling of dataset splits for robust training, validation, and testing workflows.
