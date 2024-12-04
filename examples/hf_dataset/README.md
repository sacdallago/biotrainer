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

This setup allows seamless integration of HuggingFace datasets without local files, handling splits automatically for training, validation, and testing.