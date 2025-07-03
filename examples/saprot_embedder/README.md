# saprot_embedder example

This example shows how to use biotrainer with a structure-aware embedder called
[SaProt](https://github.com/westlake-repl/SaProt) with the `sequence_to_class` protocol.
It predicts a class for every sequence in the `sequences.fasta` file.

The proteins are represented with both sequence and structural vocabulary,
so the sequence appears as `MdEvVpQpLrVyQdYaKv` for example. The minor case represents the conformation of
the corresponding residue.
[Here](https://github.com/westlake-repl/SaProt?tab=readme-ov-file#Convert-protein-structure-into-structure-aware-sequence)
you can find, how these structure-aware sequences are created.

Execute the example (from the base directory):
```bash
biotrainer train --config examples/saprot_embedder/config.yml
```
