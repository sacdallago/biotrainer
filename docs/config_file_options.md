# Configuration File Options

*Biotrainer* supports a lot of different configuration options. This file gives an overview about all of them
and explains, which protocol needs which input files and which options are mandatory.
(| means the exclusive OR, so choose one of the available options)

## General options

Choose a **protocol** that handles, how your provided data is interpreted:
```yaml
protocol: residue_to_class | residues_to_class | sequence_to_class | sequence_to_value
```

The seed can also be changed (any positive integer):
```yaml
seed: 1234  # Default: 42
```

You can also manually select a device for training:
```yaml
device: cpu | cuda  # Default: Uses cuda if cuda is available, otherwise cpu
```

After all training is done, the model is evaluated on the test set. The predictions it makes can be stored in the
output file. This behaviour is disabled by default, because the file can get very long for large datasets.
```yaml
save_test_predictions: True | False  # Default: False
```

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

## Embeddings

In *biotrainer*, it is possible to calculate embeddings automatically using *bio_embeddings*. To do this, only
a valid embedder name has to be provided:
```yaml
embedder_name: prottrans_t5_xl_u50 | esm | esm1b | seqvec | fasttext | word2vec | one_hot_encoding | ...
```
Take a look at [bio_embeddings](https://github.com/sacdallago/bio_embeddings/) to find out about all the available
embedding methods. 
A list of all current embedding config options can be found, for example, 
[in this file](https://github.com/sacdallago/bio_embeddings/blob/efb9801f0de9b9d51d19b741088763a7d2d0c3a2/bio_embeddings/embed/pipeline.py#L253). 

It is, furthermore, also possible to provide a custom embeddings file in h5 format (take a look at the 
[examples folder](../examples/h5file_enhancement/) for more information). Please also have a look at the 
[data standardization](data_standardization.md#embeddings) for the specification requirements of your embeddings.

Either provide a local file:
```yaml
embeddings_file: path/to/embeddings.h5
```
You can also download your embeddings directly from a URL:
```yaml
embeddings_file: ftp://examples.com/embeddings.h5
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

Specify the loss:
```yaml
loss_choice: cross_entropy_loss | mean_squared_error
```
Note that *mean_squared_error* can only be applied to regression tasks, i.e. *x_to_value*.

For classification tasks, the loss can also be calculated with class weights:
```yaml
use_class_weights: True | False  # Default: False
```

## Training parameters

This section describes the available training and dataloader parameters.

You should declare the maximum number of epochs you want to train your model (any positive integer):
```yaml
number_epochs: 20  # Default: 200
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

## Special training modes

On clusters for example, training can get interrupted for a numerous reasons. The implemented auto_resume mode 
makes it possible to re-submit your job without changing anything in the configuration. It will automatically search
for the latest available checkpoint at the default directory path. This behaviour is activated by default, but
you can switch it off if necessary:
```yaml
auto_resume: True | False  # Default: True
```

If you are using an already pretrained model and want to continue to train it for more epochs, you can use the 
following option:
```yaml
pretrained_model: path/to/model_checkpoint.pt
```
Biotrainer will now run until early stop was triggered, 
or `num_epochs - num_pretrained_epochs` (from the model state dict) is reached.

**Note that `pretrained_model` and `auto_resume` options are incompatible.**
`auto_resume` should be used in case one needs to restart the training job multiple times.
`pretrained_model` if one wants to continue to train a specific model.