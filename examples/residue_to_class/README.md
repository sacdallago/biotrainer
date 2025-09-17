# residue_to_class example

This example shows how to use the residue_to_class protocol.
An `input.fasta` file with the amino acid sequences and the per-residue labels (`TARGET=`) has to be provided.
For more information, see [data standardization](../../docs/data_standardization.md#residue_to_class). 

Additionally, in this example the `use_class_weights: True` flag is set, 
which can also be used to get a quick overview about the class distribution in your dataset from the console logs.

Execute the example (from the base directory):
```bash
biotrainer train --config examples/residue_to_class/config.yml
```
