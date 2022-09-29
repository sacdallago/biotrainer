# First steps with biotrainer

In this step-by-step tutorial, we want to show you how to use *biotrainer* from scratch. You will be provided with all
necessary files to follow the tutorial on your own. 

## Prediction task

We will train a model for [conservation prediction](https://link.springer.com/article/10.1007/s00439-021-02411-y).
This means that for every residue in a protein sequence, a *conservation score* ranging from *0 to 9* is predicted.
The higher the score, the more conserved the residue was during evolution. Thus, conservation is a good indicator
if a mutation on this residue (single amino acid variant, *SAV*) has an effect on the functional properties of the 
protein. A possible application of these predictions could be to analyze likely mutations of virus proteins, which
was strongly demanded for the Sars-Cov-2 virus in particular.

Thus, our problem belongs to the [residue_to_class protocol](data_standardization.md#residue_to_class).

## Installation of biotrainer

Make sure that you have *biotrainer* [installed](../README.md#installation) with *extra* `bio-embeddings`:
```bash
poetry install --extras "bio-embeddings"
```

## Downloading the dataset

Next, we need to download our dataset. The conservation dataset we use is included in the 
[FLIP repository](https://github.com/J-SNACKKB/FLIP), which contains multiple standardized datasets and benchmarks 
for relevant protein engineering and prediction tasks. In the repository, select 
[splits -> conservation](https://github.com/J-SNACKKB/FLIP/tree/main/splits/conservation) 
and download the *splits.zip* file.
Create a new directory in your biotrainer folder and extract the zip file there. 
You should now have access to two files: 
1. **sequences.fasta**: This file contains the sequence id, e.g. `3p6z-C` and the amino acid sequence.
By using the sequence id, you can look up your protein at [PDB](https://www.rcsb.org/structure/3P6Z) for example.
The letter after the dash indicates the 
[chain](https://biology.stackexchange.com/questions/37495/what-is-chain-identifier-in-pdb).
2. **sampled.fasta**: This is our labels file, which contains the conservation scores for each residue in the amino
acid sequences. Of course, the corresponding sequence ids have to match here. Additionally, it also includes
the dataset annotations, which divide the sequences into `train/validation/test` sets. As the name of the file suggests,
splits have been generated randomly, with a distribution of about 90% train, 5% validation, 5% test.

## Creating the configuration file

Now that we have the dataset files ready, we need to specify a config file in `.yaml` format.

Create and save the following file in the directory you created before:
```yaml
# config.yml
sequence_file: splits/sequences.fasta
labels_file: splits/sampled.fasta
protocol: residue_to_class
model_choice: CNN
optimizer_choice: adam
loss_choice: cross_entropy_loss
num_epochs: 30
patience: 20
use_class_weights: True
learning_rate: 1e-3
batch_size: 128
embedder_name: one_hot_encoding
```

We are using pretty standard parameters here to train a CNN. We increased the `patience` a bit, to be more indulgent
with the training process in the tutorial. If you do not have a GPU available, consider using a smaller number
of epochs if the training process takes too long.

The most interesting parameter is the `embedder_name: one_hot_encoding` one. This means that we automatically convert
our sequences to one-hot encoded vectors, where the actually present amino acid has value `1`, while all the others
have value `0`. This is a pretty minimal embedding, but it is sufficient for the purpose of this tutorial. Feel free
to test other [embedding methods](config_file_options.md#embeddings) as well! Just keep in mind that some of them
require a lot of memory and GPU power to compute.

## Running the training

We are finally able to train our model! Execute the following command, using the directory you created:
```bash
poetry run python3 run-biotrainer.py your-directory/config.yml
```


