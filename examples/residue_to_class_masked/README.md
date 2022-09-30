# residue_to_class - Masked example

This example shows how to use a mask file together with sequences and labels files for the residue_to_class protocol.
The mask file must contain the same sequence ids as the other two files and the number of residues must be a 1:1 match
across files.

The mask file option is set via:
```yaml
mask_file: mask.fasta
```

Execute the example (from the base directory):
```bash
poetry run python3 run-biotrainer.py examples/residue_to_class_masked/config.yml
```
