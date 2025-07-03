# protein_protein_interaction example

This example shows how to use the sequence_to_class protocol for protein interactions. 
In this case, only one file (with an arbitrary name) has to be provided:
`interactions.fasta` contains the ids of the interacting proteins (`>ID INTERACTOR=ID2`), their associated amino
acid sequences and the `TARGET=0/1` to indicate, if they interact or not. Split annotations are given via
`SET=train/val/test`.

Execute the example (from the base directory):
```bash
biotrainer train --config examples/protein_protein_interaction/config.yml
```
