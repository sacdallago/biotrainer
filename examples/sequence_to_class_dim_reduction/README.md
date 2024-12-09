# sequence_to_class_dim_reduction example

This example shows how to use the sequence_to_class protocol with embeddings dimensionality reduction. It predicts a class for every sequence with reduced dimension in the
sequences.fasta file. Class labels and dataset annotations are also stored in the sequences.fasta file 
for this protocol (see [data standardization](../../docs/data_standardization.md#sequence_to_class)).

Additionally, in this example the `use_class_weights: True` flag is set, 
which can also be used to get a quick overview about the class distribution in your dataset from the console logs.

Execute the example (from the base directory):
```bash
poetry run python3 run-biotrainer.py examples/sequence_to_class_dim_reduction/config.yml
```
