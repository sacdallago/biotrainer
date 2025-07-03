# continue model training example

If you have an already pre-trained model or want to continue to train a model from a checkpoint manually,
this example shows how to achieve that. The example uses the sequence_to_class protocol.

The [sample checkpoint](output/FNN/one_hot_encoding/sample_checkpoint.pt) in the output directory gets loaded 
and continues training until the maximum number of epochs has been reached or early stop was triggered.

The configuration option:
```yaml
pretrained_model: output/FNN/one_hot_encoding/sample_checkpoint.pt
```

**Note that `pretrained_model` and `auto_resume` options are incompatible.**
`auto_resume` should be used in case one needs to restart the training job multiple times.
`pretrained_model` if one wants to continue to train a specific model.

Execute the example (from the base directory):
```bash
biotrainer train --config examples/continue_model_training/config.yml
```

