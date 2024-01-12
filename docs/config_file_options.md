# Configuration File Options

*Biotrainer* supports a lot of different configuration options. This file gives an overview about all of them
and explains, which protocol needs which input files and which options are mandatory.
(| means the exclusive OR, so choose one of the available options)

## General options

Choose a **protocol** that handles, how your provided data is interpreted:
```yaml
protocol: residue_to_class | residues_to_class | sequence_to_class | sequence_to_value
```

To use interactions for your sequences, you need to indicate how embeddings of interacting sequences should be
combined:
```yaml
interaction: multiply | concat  # Default: None
```
`multiply` does an element-wise multiplication of the two embeddings, so `torch.mul(emb1, emb2)`.
`concat` concatenates both embeddings, so `torch.concat([emb1, emb2])`.

The seed can also be changed (any positive integer):
```yaml
seed: 1234  # Default: 42
```

You can also manually select a device for training:
```yaml
device: cpu | cuda | cuda:0 | cuda:1 .. # Default: Uses cuda if cuda is available, otherwise cpu
```

The sequence ids associated with the generated or annotated splits can be saved in the output file.
Furthermore, after all training is done, the (best) model is evaluated on the test set. 
The predictions it makes are stored in the output file, if the following flag is set to `True`. 
This behaviour is disabled by default, because the file can get very long for large datasets.
```yaml
save_split_ids: True | False  # Default: False
```

Sometimes, your sequence or embeddings file might contain more or less sequences than your corresponding labels 
file for a residue-level prediction task. 
If this is intended, you can tell *biotrainer* to simply ignore those redundant sequences:
```yaml
ignore_file_inconsistencies: True | False  # Default: False
```
If `True`, *biotrainer* will also not check if you provided at least one sample for each split set. 

A concrete output directory can be specified:
```yaml
output_dir: path/to/output_dir  # Default: path/to/config/output
```

Biotrainer automatically performs sanity checks on your test results. The checks are handled by the 
`validations` module. For example, a warning will be logged if the model predicts only one unique value for every
entry in the test set. The module also automatically calculates baselines (e.g. predicting only `1` or `0` for all 
entries in the test set) with the respective metrics for suitable protocols. All baselines are also stored in the
output file.
The behaviour is enabled by default but can be switched off:
```yaml
sanity_check: True | False  # Default: True
```
*Keep in mind that this can only be a first insight into the predictive capabilities of your model,
no warnings in the logs do not imply that the results make (biological) sense!*

## Training data (protocol specific)

Depending on the protocol that fits to your dataset, you have to provide different input files. For the data standards
of these files, please refer to the [data standardization](data_standardization.md) document.

A sequence file has to be provided for every protocol:
```yaml
sequences_file: path/to/sequences.fasta
```

For the **residue_to_class** protocol, a separate labels file also has to be provided:
```yaml
labels_file: path/to/labels.fasta
```

The **residue_to_class** and **residues_to_class** protocols also support a mask file, if some residues should be masked
during training:
```yaml
mask_file: path/to/mask.fasta
```
The mask file must provide a value of `0` or `1` for each amino acid of every sequence. Unresolved residues are not
included in class weight calculation.

## Embeddings

In *biotrainer*, it is possible to calculate embeddings automatically. Since of version 0.8.0, *bio_embeddings* has
been deprecated as package for embeddings calculation. Instead, *biotrainer* now supports embedding calculation
directly via [huggingface transformers](https://huggingface.co/docs/transformers/index).
The following embedders have been successfully tested with *biotrainer*:
```yaml
embedder_name: Rostlab/prot_t5_xl_uniref50 | ElnaggarLab/ankh-large | Rostlab/prot_t5_xl_bfd
```

Transformer embedders provide the ability to use half-precision (float16) mode:
```yaml
use_half_precision: True | False # Default: False
```
*Note that use_half_precision mode is not compatible with embedding on the CPU 
([see this GitHub Issue](https://github.com/huggingface/transformers/issues/11546)).*

To compute baselines, there are also predefined embedders directly included in *biotrainer*:
```yaml
embedder_name: one_hot_encoding
```

If you want to use your own embedder directly in *biotrainer*, you can provide it as a python script. 
You only need to extend the `CustomEmbedder` class from the `embeddings` module. Please take a look at the
example implementations of `esm-2` in the [custom_embedder example](../examples/custom_embedder/).
Then, use the `embedder_name` option to declare the path to your script:
```yaml
embedder_name: your_script.py  # The script path must be relative to the configuration file or absolute.
```
**Security measures are in place to prevent arbitrary code from execution. 
Still, this might introduce a security risk when running biotrainer as a remote service. 
Downloading of any custom_embedder source file during execution is therefore disabled. Please only run scripts
from sources you trust.** 

It is also possible to provide a custom embeddings file in h5 format (take a look at the 
[examples folder](../examples/custom_embeddings/) for more information). Please also have a look at the 
[data standardization](data_standardization.md#embeddings) for the specification requirements of your embeddings.

Either provide a local file:
```yaml
embeddings_file: path/to/embeddings.h5
```
You can also download your embeddings directly from a URL:
```yaml
embeddings_file: ftp://examples.com/embeddings.h5  # Supports http, https, ftp
```
The file will be downloaded and stored in the path of your config file with prefix "downloaded_".

**Note that *embedder_name* and *embeddings_file* are mutually exclusive. In case you provide your own embeddings,
the experiment directory will be called *custom_embeddings*.**

## Model parameters

There are multiple options available to specify the model you want to train.

At first, choose the model architecture to be used:
```yaml
model_choice: FNN | CNN | LogReg | LightAttention  # Default: CNN
```
<details><summary>The available models depend on your chosen protocol.</summary>
<code>
'residue_to_class': {
    CNN,
    FNN,
    LogReg
},
'residues_to_class': {
    LightAttention 
}
'sequence_to_class': {
    FNN,
    LogReg
},
'sequence_to_value': {
    FNN,
    LogReg
}
</code>
</details>

Specify an optimizer:
```yaml
optimizer_choice: adam  # Default: adam
```
and the learning rate (any positive float):
```yaml
learning_rate: 1e-4  # Default: 1e-3
```

Set the dropout rate for dropout layers if desired:
```yaml
dropout_rate: 0.5  # Default: 0.25
```
*Note that some models, including the implemented logistic regression model, do not support dropout layers.*

Specify the loss:
```yaml
loss_choice: cross_entropy_loss | mean_squared_error
```
Note that *mean_squared_error* can only be applied to regression tasks, i.e. *x_to_value*.

For classification tasks, the loss can also be calculated with class weights:
```yaml
use_class_weights: True | False  # Default: False
```
Class weights only take the training data into account for calculation. Masked residues, denoted in a mask file,
are not included in class weight calculation as well.

PyTorch 2.0 introduced [torch compile](https://pytorch.org/docs/stable/generated/torch.compile.html) for faster
and more efficient training. Since biotrainer v0.8.1, this can be enabled for model training via:
```yaml
disable_pytorch_compile: False  # Default: True
```
*The behaviour is disabled by default for backwards compatibility and stability.*

## Training parameters

This section describes the available training and dataloader parameters.

You should declare the maximum number of epochs you want to train your model (any positive integer):
```yaml
num_epochs: 20  # Default: 200
```

The build-in solver has an early-stop mechanism that prevents the model from over-fitting if the validation loss 
increases for too long. It can be controlled by providing a patience value (any positive integer):
```yaml
patience: 20  # Default: 10
```
To provide a threshold that indicates, if the patience should decrease, an epsilon value can be specified (any positive
float):
```yaml
epsilon: 1e-3  # Default: 0.001 
```
<details><summary>Early stop mechanism explanation:</summary>
If the current loss is smaller than the previous minimum loss minus the epsilon threshold, a stop count is reset to 
the patience value indicated above. 
Otherwise, if it is already 0, early stop is triggered and the best previous model loaded.
If it is still above 0, patience gets decreased by one.
Expressed in code:
<code>
    
    def _early_stop(self, current_loss: float, epoch: int) -> bool:
        if current_loss < (self._min_loss - self.epsilon):
            self._min_loss = current_loss
            self._stop_count = self.patience

            # Save best model (overwrite if necessary)
            self._save_checkpoint(epoch)
            return False
        else:
            if self._stop_count == 0:
                # Reload best model
                self.load_checkpoint()
                return True
            else:
                self._stop_count = self._stop_count - 1
                return False
</code>
</details>

It is also possible to change the batch size (any positive integer):
```yaml
batch_size: 32  # Default: 128
```

The dataloader can also shuffle the dataset on each epoch:
```yaml
shuffle: True | False  # Default: True
```

## Cross Validation

### Default

By default, *biotrainer* will use the set annotations provided via the sequence annotations. Separating the 
data this way into train, validation and test set is commonly known as **hold-out cross validation**.
```yaml
cross_validation_config:
  method: hold_out  # Default: hold_out
```
This is the default option and specifying it in the config file is optional.

Additionally, *biotrainer* supports a set of other cross validation methods, which can be configured in the 
configuration file, as shown in the next sections. We assume knowledge of the methods and only provide the available
configuration options.
[This article](https://neptune.ai/blog/cross-validation-in-machine-learning-how-to-do-it-right) provides a more
in-depth description of the implemented cross validation methods.

### k-fold Cross Validation

When using k-fold cross validation, sequences annotated with either "SET=train" or "VALIDATION=True" are combined
to create the k splits. *Biotrainer* supports **repeated**, **stratified** and **nested** k-fold cross validation
(and combinations of these techniques).
```yaml
cross_validation_config:
  method: k_fold
  k: 2  # k >= 2
  stratified: True  # Default: False
  repeat: 3  # Default: 1
```
The distribution of train to validation samples in the splits is `(k-1):1`, 
so three times more train than validation samples for `k: 4`.

The `stratified` option can also be used for regression tasks. In this case, the continuous values are converted
to bins to calculate the stratified splits.

To use **nested** k-fold cross validation, hyperparameters which should be optimized with the nested splits must
be specified in the config file. This can be done by using lists explicitly, using pythonic list comprehensions
or a range expression. The latter two have to be provided as string literals.
The following example shows all three options:
```yaml
use_class_weights: [True, False]  # Explicit list
learning_rate: "[10**-x for x in [2, 3, 4]]"  # List comprehension
batch_size: "range(8, 132, 4)"  # Range expression
```
All parameters that change during training can be configured this way to be included in the hyperparameter optimization
search. As search methods, **random search** and **grid search** have been implemented. 
A complete nested k-fold cross validation config could look like this:
```yaml
cross_validation_config:
  method: k_fold
  k: 3  # k >= 2
  stratified: True  # Default: False
  repeat: 1  # Default: 1
  nested: True
  nested_k: 2  # nested_k >= 2
  search_method: random_search
  n_max_evaluations_random: 3  #  n_max_evaluations_random >= 2
```

Note that the total number of trained models will be `k * nested_k * n_max_evaluations_random` for random_search
and `k * nested_k * len(possible_grid_combinations)`, so the training process can get very resource-heavy!

Overview about all config options for k-fold cross validation:
```yaml
cross_validation_config:
  method: k_fold
  k: 5  # k >= 2
  stratified: True | False # Default: False
  repeat: 1  # repeat >= 1, Default: 1
  nested: True | False  # Default: False
  nested_k: 3  # nested_k >= 2, only nested
  search_method: random_search | grid_search  # No default, but must be configured for nested k-fold cv, only nested
  n_max_evaluations_random: 3  #  n_max_evaluations_random >= 2, only for random_search, only nested
```

### Leave-p-out Cross Validation

This edge case of k-fold Cross Validation with (usually) very small validation sets is also implemented in
*biotrainer*.
Using it is quite simple:
```yaml
cross_validation_config:
  method: leave_p_out
  p: 5  # p >= 1
```

**Note that this might create a very high number of splits and is only recommended for small training and validation
sets!**

For both cross validation methods, you can also declare which metric to use to choose the best model
(or the best hyperparameter combination for nested k-fold cross validation):
```yaml
cross_validation_config:
  method: leave_p_out
  choose_by: loss | accuracy | precision | recall | rmse | ... # Default: loss 
  p: 5  # p >= 1
```

## Special training modes

On clusters for example, training can get interrupted for a numerous reasons. The implemented auto_resume mode 
makes it possible to re-submit your job without changing anything in the configuration. It will automatically search
for the latest available checkpoint at the default directory path. This behaviour is de-activated by default, but
you can switch it on if necessary:
```yaml
auto_resume: True | False  # Default: True
```
This works for any cross validation method.
**Due to randomness in the computation (especially when using GPU or random layers in the network like dropout),
results might differ between originally trained checkpoints and resumed training!**

If you are using an already pretrained model and want to continue to train it for more epochs, you can use the 
following option:
```yaml
pretrained_model: path/to/model_checkpoint.pt
```
**Note that `pretrained_model` only works in combination with `hold_out` cross validation!**

*Biotrainer* will now run until early stop was triggered, 
or `num_epochs - num_pretrained_epochs` (from the model state dict) is reached.

**Note that `pretrained_model` and `auto_resume` options are incompatible.**
`auto_resume` should be used in case one needs to restart the training job multiple times.
`pretrained_model` if one wants to continue to train a specific model.


Sometimes it might be useful to check your employed setup and architecture only on a small sub-sample of your 
total dataset. This can also be automatically done for you in *biotrainer*:
```yaml
limited_sample_size: 100  # Default: -1, must be > 0 to be applied
```
Note that this value is applied only to the train dataset and embedding calculation is currently
done for all sequences!