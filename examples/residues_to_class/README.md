# residues_to_class example

The example shown here includes a sequences.fasta file and a config.yaml file, that declare a training process
for the residues_to_class protocol. This protocol uses per-residue embeddings, but predicts features on the sequence
level. It employs the 
[LightAttention](https://github.com/HannesStark/protein-localization/blob/master/models/light_attention.py#L5) model.

Execute the example (from the base directory):
```bash
biotrainer train --config examples/residues_to_class/config.yml
```
