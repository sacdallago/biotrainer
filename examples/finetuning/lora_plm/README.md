# LoRA Finetuning for a protein language model example

This example shows how to use [LoRA finetuning](https://doi.org/10.48550/arXiv.2106.09685) to finetune a pLM on a given set of proteins. To achieve this,
we use the `residue_to_class` protocol and employ a CNN as downstream decoder model.
We set the `TARGET` values to the given protein sequence and enable the `random_masking` config option for 
automated token masking.

Execute the example (from the base directory):
```bash
biotrainer train --config examples/finetuning/lora_plm/config.yml
```
