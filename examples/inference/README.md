# inference example

These examples show how to use a model trained with *biotrainer* for inference. It is done by using the
*Inferencer* rather than the *Solver* class. A pretrained model, along with the necessary files, can be found
in the [residue_to_class example](../residue_to_class/). The notebook shows how to use them to predict per-residue
classes for any new sequences.

Please also refer to the `cli` [predict command](../docs/cli.md).

Note that you need to have `jupyter` installed for the notebook:
```shell
uv pip install -e ".[jupyter]"
```