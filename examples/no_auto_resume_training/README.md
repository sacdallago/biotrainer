# no auto resume training example

This example was created to show the default behaviour of *biotrainer*: By default, training gets re-started from
an existing checkpoint, if this checkpoint is found in the default directory in the *output* folder.
Especially on servers or clusters, where it often can happen that training gets interrupted, this enables the user
to easily re-submit the job without having to change anything about the configuration file or checkpoints.
Training then gets re-started from the latest checkpoint and runs until the specified number of epochs is reached
or early stop was triggered.

However, if this behaviour is not desired, it can be turned off as shown here:
```yaml
auto_resume: False
```

**Note that `pretrained_model` and `auto_resume` options are incompatible.**
`auto_resume` should be used in case one needs to restart the training job multiple times.
`pretrained_model` if one wants to continue to train a specific model.

Execute the example (from the base directory):
```bash
biotrainer train --config examples/no_auto_resume_training/config.yml
```