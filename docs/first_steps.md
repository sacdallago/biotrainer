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

Make sure you have [uv](https://github.com/astral-sh/uv) installed.

Now you can [install](../README.md) *biotrainer* via `uv`:
```shell
# Create and activate a virtual environment
uv venv
source .venv/bin/activate  # On Unix/macOS
# OR
.venv\Scripts\activate  # On Windows

# Basic installation of biotrainer
uv pip install -e .
```

## The Dataset

Next, we need to download our dataset. The conservation dataset we use is included in the 
[FLIP repository](https://github.com/J-SNACKKB/FLIP), which contains multiple standardized datasets and benchmarks 
for relevant protein engineering and prediction tasks.

We are using the following files: 
1. [sequences.fasta](http://data.bioembeddings.com/public/FLIP/fasta/conservation/sequences.fasta): 
This file contains the sequence id, e.g. `3p6z-C` and the amino acid sequence.
By using the sequence id, you can look up your protein at [PDB](https://www.rcsb.org/structure/3P6Z) for example.
The letter after the dash indicates the 
[chain](https://biology.stackexchange.com/questions/37495/what-is-chain-identifier-in-pdb).
2. [sampled.fasta](http://data.bioembeddings.com/public/FLIP/fasta/conservation/sampled.fasta): 
This is our labels file, which contains the conservation scores for each residue in the amino
acid sequences. Of course, the corresponding sequence ids have to match here. Additionally, it also includes
the dataset annotations, which divide the sequences into `train/validation/test` sets. As the name of the file suggests,
splits have been generated randomly, with a distribution of about 90% train, 5% validation, 5% test.

We combine the two files to our input file:
```shell
biotrainer convert --sequence-file sequences.fasta --labels-file sampled.fasta --skip_inconsistencies
```

## Creating the configuration file

Next, we need to specify a config file in `.yaml` format. You need to specify the path to the combined dataset there
(or just copy the file in the same directory as the config file).

Create and save the following file in the new directory:
```yaml
# config.yaml
input_file: path/to/converted.fasta
protocol: residue_to_class
model_choice: CNN
device: cpu
optimizer_choice: adam
learning_rate: 1e-3
loss_choice: cross_entropy_loss
num_epochs: 30
batch_size: 128
ignore_file_inconsistencies: True
cross_validation_config:
  method: hold_out
embedder_name: one_hot_encoding
```

We are using pretty standard parameters to train a CNN. 
If you do have a GPU available, you might want to increase the num_epochs parameter and see, how the model behaves.

An important parameter is the `embedder_name: one_hot_encoding` one. This means that we automatically convert
our sequences to one-hot encoded vectors, where the actually present amino acid has value `1`, while all the others
have value `0`. This is a pretty minimal embedding, but it is sufficient for the purpose of this tutorial. Feel free
to test other [embedding methods](config_file_options.md#embeddings) as well! Just keep in mind that some of them
require a lot of memory and GPU power to compute.

## Running the training

We are finally able to train our model! Execute the following command, using the directory you created:
```shell
biotrainer train --config your-directory/config.yaml
```

After a while, the training should be finished. You can find the best model checkpoint and a summary *out.yaml*
file in an *output* subdirectory in the folder of your config file.

## Where to go from here

You finished your first run of *biotrainer*, thank you for using our software!
In our test run, we got an accuracy of about 0.20 on the test set. We are confident that you could improve that
result by using different model and training parameters or a more sophisticated embedding method!

Of course, you can now start to use *biotrainer* on your own protein data. Or you could employ your own embedder
and use its embeddings for training. You can also find plenty of other examples [here](../examples), including 
[tutorial notebooks](../examples/tutorials) that show you how to use biotrainer from python code directly.
If you have any feedback, please do not hesitate to get in touch, e.g. by creating an [issue](https://github.com/sacdallago/biotrainer/issues).
