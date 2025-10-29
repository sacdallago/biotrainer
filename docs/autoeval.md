# Autoeval Module

The biotrainer `autoeval` module allows automatically evaluating an embedder model on downstream prediction tasks.
This can give a better impression of the model performance, and if it is actually creating useful embeddings.

## Downstream Tasks

All relevant files for the downstream tasks are downloaded before the evaluation to
`/{user_cache_dir}/biotrainer/autoeval`.

### PBC

The `autoeval` module provides direct support of the [PBC](https://github.com/Rostlab/pbc) framework, namely for the
supervised tasks.

Quick description of each task, you can find more information in the [PBC Readme](https://github.com/Rostlab/pbc):

* **scl**: Predict protein subcellular localization for mixed human and non-human proteins.
* **secondary_structure**: Predict secondary structure of proteins (includes multiple test sets from CASP and the ProtT5
  paper).

### FLIP

**WARNING: FLIP datasets are currently still supported but not actively maintained.
Please refer to the [PBC](https://github.com/Rostlab/pbc) framework instead.**

The `autoeval` module provides a curated subset of datasets in [FLIP](https://github.com/J-SNACKKB/FLIP).
The tasks have been chosen to be as hard as possible for a prediction model.
We provide a `flip_conversion_script.py` in the module that shows, how the datasets have been preprocessed
and curated.

Quick description of each task, you can find more information in
the [split descriptions](https://github.com/J-SNACKKB/FLIP/tree/main/splits):

* **aav**: Predict adeno-associated virus fitness.
    * *low_vs_high*: Training on low fitness values, testing on high fitness values
    * *two_vs_many*: Training on two mutations max, testing on many mutations
* **bind**: Predict binding sites of ligands for protein residues.
    * from_publication: Dataset split used in [this](https://doi.org/10.1038/s41598-021-03431-4) publication
* **gb1**: Predict a binding score for each variant of the GB1 protein.
    * *low_vs_high*: Training on low fitness values, testing on high fitness values
    * *two_vs_many*: Training on two mutations max, testing on others
* **meltome**: Predict meltome temperature of proteins.
    * *mixed_split*: Mixed human and non-human proteins
* **scl**: Predict protein subcellular localization.
    * *mixed_hard*: Mixed human and non-human proteins
* **secondary_structure**: Predict secondary structure of proteins
    * *sampled*: Only available FLIP split

## How to use

### CLI

The `autoeval` pipeline can be run directly through the cli:

```shell
biotrainer autoeval --embedder-name embedder_name --framework framework_name [--min-seq-length min_length] [--max-seq-length max_length] [--use-half-precision]
```

### Script

You can also integrate `autoeval` into your scripts or training pipelines:

```python
from typing import Iterator

import torch
import numpy as np
from tqdm import tqdm

from biotrainer.autoeval import autoeval_pipeline
from biotrainer.utilities import seed_all


class ExampleRandomEmbedder:
    def __init__(self, embedding_dim: int = 21):
        self.embedding_dim = embedding_dim
        # Pre-generate random state for faster random number generation
        self.rng = np.random.default_rng()

    def embed_per_residue(self, sequences: Iterator[str]):
        for sequence in tqdm(sequences):
            # Generate all embeddings at once using rng
            embedding = self.rng.random((len(sequence), self.embedding_dim), dtype=np.float32)
            yield sequence, torch.tensor(embedding)

    def embed_per_sequence(self, sequences: Iterator[str]):
        for sequence in tqdm(sequences):
            # Generate single embedding using rng
            embedding = self.rng.random(self.embedding_dim, dtype=np.float32)
            yield sequence, torch.tensor(embedding)


seed_all(42)

embedder = ExampleRandomEmbedder()

for progress in autoeval_pipeline(embedder_name="example_random_embedder",
                                  framework="pbc",
                                  custom_embedding_function_per_residue=lambda seqs: embedder.embed_per_residue(
                                      seqs),
                                  custom_embedding_function_per_sequence=lambda
                                          seqs: embedder.embed_per_sequence(seqs),
                                  ):
    print(progress)
```

Some deeper explanations:

* The `autoeval_pipeline` function is a generator function that yields `AutoEvalProgress` objects to track progress.
  Therefore, you must "do something" with the pipeline return values, in order to execute the pipeline. Simply calling
  `autoeval_pipeline(...)` will not run the pipeline, see [python generators](https://wiki.python.org/moin/Generators).
* The custom embedding functions take an iterable of sequences as strings and need to yield a sequence and the according
  embedding. This is to make sure that the correct embedding is assigned to the respective sequence.
* All embeddings are calculated at once at the beginning of the training, to avoid duplicated embedding.

## Report

After all tasks have been successfully finished, a report is created in the output directory. All metadata and
model results are tracked there.

Example:

```json
{
  "embedder_name": "your_embedder_name",
  // Embedder name
  "training_date": "2025-07-02",
  // Training date
  "min_seq_len": 0,
  // Minimum sequence length
  "max_seq_len": 2000,
  // Maximum sequence length
  "results": {
    "FLIP-aav-two_vs_many": {
      "config": {
      },
      "database_type": "Protein",
      "derived_values": {
      },
      "training_results": {
      },
      "test_results": {
        "test": {
          "metrics": {
            "loss": 12.494811077213766,
            "mse": 12.501708030700684,
            "rmse": 3.5357754230499268,
            "spearmans-corr-coeff": 0.0014577994588762522
          },
          "bootstrapping": {
            "results": {
              "mse": {
                "mean": 12.5078125,
                "error": 0.099365234375
              },
              "rmse": {
                "mean": 3.537109375,
                "error": 0.01436614990234375
              },
              "spearmans-corr-coeff": {
                "mean": 0.0007982254028320312,
                "error": 0.00795745849609375
              }
            },
            "iterations": 30,
            "sample_size": 50767,
            "confidence_level": 0.05
          },
          "test_baselines": {
          },
          "predictions": {}
        }
      }
    }
    // OTHER TASKS
  }
}
```

### Visualization and Leaderboard

You can visualize your results and compare against other embedder
models using [biocentral](https://app.biocentral.cloud). Simply load the report from file in the pLM Evaluation module.
