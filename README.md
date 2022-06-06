# Biotrainer

## Basic usage

### Installation

1. Make sure you have [poetry](https://python-poetry.org/) installed: 
```bash
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/install-poetry.py | python3 - --version 1.1.13
```

2. Install dependencies and biotrainer via `poetry`:
```bash
# In the base directory:
poetry install
# Optional: Add bio-embeddings to compute embeddings
poetry install --extras "bio-embeddings"
```

### Running

```bash
cd examples/residue_to_class
poetry run biotrainer config.yml
```

You can also use the provided `run-biotrainer.py` file for development and debugging (you might want to set up your 
IDE to directly execute run-biotrainer.py with the provided virtual environment):
```bash
# Residue -> Class
poetry run python3 run-biotrainer.py examples/residue_to_class/config.yml
# Sequence -> Class
poetry run python3 biotrainer.py examples/sequence_to_class/config.yml
```

### Docker

```bash
# Build
docker build -t biotrainer .
# Run
docker run --rm \
    -v "$(pwd)/examples/docker":/mnt \
    -v bio_embeddings_weights_cache:/root/.cache/bio_embeddings \
    -u $(id -u ${USER}):$(id -g ${USER}) \
    biotrainer:latest /mnt/config.yml
```

Output can be found afterwards in the directory of the provided configuration file.

## Available protocols

```text
D=embedding dimension (e.g. 1024)
B=batch dimension (e.g. 30)
L=sequence dimension (e.g. 350)
C=number of classes (e.g. 13)

- residue_to_class --> Predict a class C for each residue encoded in D dimensions in a sequence of length L. Input BxLxD --> output BxLxC
- sequence_to_class --> Predict a class C for each sequence encoded in a fixed dimension D. Input BxD --> output BxC

Work in Progress:
- residues_to_class --> Predict a class C for all residues encoded in D dimensions in a sequence of length L. Input BxLxD --> output BxC
```

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
embeddings_file_path: /path/to/embeddings.h5 # optional, if defined will use 'embedder_name' to name experiment
```

### Available embedding methods
Take a look at [bio_embeddings](https://github.com/sacdallago/bio_embeddings/) to find out about all the available
embedding methods. 
A list of embedding config options can be found, for example, [in this file](https://github.com/sacdallago/bio_embeddings/blob/develop/bio_embeddings/embed/pipeline.py). 
