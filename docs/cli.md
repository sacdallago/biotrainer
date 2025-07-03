# Command Line Interface

*Biotrainer* provides several command-line interface (CLI) commands for different tasks. This document provides an
overview of all available commands and their options.

## Training

The `train` command is used to start a training run with a specified configuration:

```shell
biotrainer train --config path/to/config.yml
```

This command accepts either a path to a YAML configuration file or a configuration dictionary and executes the training
pipeline according to the specified parameters.

## Prediction

The `predict` command allows you to use a trained model for making predictions:

```shell
biotrainer predict --training-output-file path/to/training_output.json --model-input input_sequences [--save-embeddings]
```

### Parameters:

- `--training-output-file`: Path to the training output file generated during model training
- `--model-input`: Either a path to a FASTA file or a comma-separated list of sequences
- `--save-embeddings`: Optional flag to save the computed embeddings (Default: False)

The command will output predictions for each input sequence, displaying the sequence ID and its corresponding
prediction.

### Example

```shell
biotrainer predict --training-output-file examples/residue_to_class/output/out.yml --model-input "HMMHM","MAHM"
```

## Format Conversion

The `convert` command helps to convert the deprecated three-way biotrainer files (sequence, labels, masks) to 
the new single file (`input_file`) input:

```shell
biotrainer convert --sequence-file sequences.fasta [--labels-file labels.txt] [--masks-file masks.txt] [--converted-file output.fasta] [--target-format fasta] [--skip_inconsistencies]
```

### Parameters:

- `--sequence-file`: Input sequence file
- `--labels-file`: Optional file containing labels
- `--masks-file`: Optional file containing masks
- `--converted-file`: Output file name (Default: "converted.fasta")
- `--target-format`: Target format for conversion (Default: "fasta", "csv" is planned but not currently supported)
- `--skip_inconsistencies`: Whether to skip inconsistent sequences from multiple files (e.g. no label for a sequence, Default: False)

## Auto-Evaluation

The `autoeval` command performs automatic evaluation of embedder model performance on a given framework of downstream
tasks:

```shell
biotrainer autoeval --embedder-name embedder_name --framework framework_name [--min-seq-length min_length] [--max-seq-length max_length] [--use-half-precision]
```

### Parameters:

- `--embedder-name`: Name of the embedder to evaluate
- `--framework`: Name of the framework to use (currently only "flip" is supported)
- `--min-seq-length`: Minimum sequence length to consider (Default: 0)
- `--max-seq-length`: Maximum sequence length to consider (Default: 2000)
- `--use-half-precision`: Whether to use half-precision computation (Default: False)

The command will print progress updates during the evaluation process.

## Important Notes

- All parameters must be explicitly set when using the CLI commands
- File paths can be provided as either relative or absolute paths
- For the `predict` command, sequences can be provided either through a FASTA file or as a comma-separated list
- The `convert` command is particularly useful for converting deprecated file formats to the current standard