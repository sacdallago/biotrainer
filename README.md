# Biotrainer

## Basic usage

*Residue-to-class*:
```bash
python biotrainer.py examples/residue_to_class/config.yml
```

*Sequence-to-class*:
```bash
python biotrainer.py examples/sequence_to_class/config.yml
```

Output can be found afterwards in the dataset directory.

## Configuration file options

```yaml
sequence_file: sequences.fasta # Specify your sequence file
labels_file: labels.fasta # Specify your label file
protocol: residue_to_class # residue_to_class | sequence_to_class : Prediction method
model_choice: CNN # CNN | ConvNeXt | FNN | LogReg : Prediction model 
optimizer_choice: adam # adam : Model optimizer
loss_choice: cross_entropy_loss # cross_entropy_loss : Model loss 
num_epochs: 200 # 1-n : Number of maximum epochs
use_class_weights: False # Balance class weights by using class sample size in the given dataset
learning_rate: 1e-3 # 0-n : Model learning rate
batch_size: 128 # 1-n : Batch size
embedder_name: prottrans_t5_xl_u50 # one_hot_encoding | word2vec | prottrans_t5_xl_u50 | ... : Sequence embedding method (see below)
```

### Available embedding methods
Take a look at [bio_embeddings](https://github.com/sacdallago/bio_embeddings/) to find out about all the available
embedding methods. 
A list of embedding config options can be found, for example, [in this file](https://github.com/sacdallago/bio_embeddings/blob/develop/bio_embeddings/embed/pipeline.py). 
