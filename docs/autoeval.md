# Autoeval Module

The biotrainer `autoeval` module allows automatically evaluating a protein language model on downstream tasks.
This can give a better impression of the model performance, and if it is actually creating useful embeddings.

## Getting Started

Please refer to the two tutorial notebooks to get started with [supervised](../examples/autoeval/plm_eval.ipynb)
and [zero-shot](../examples/autoeval/plm_eval_zeroshot.ipynb) evaluation.

## Downstream Tasks

All relevant files for the downstream tasks are downloaded before the evaluation to
`/{user_cache_dir}/biotrainer/autoeval`.

### PBC (Supervised)

The `autoeval` module provides direct support of the [PBC](https://github.com/Rostlab/pbc) framework, namely for the
supervised tasks.

Quick description of each task, you can find more information in the [PBC Readme](https://github.com/Rostlab/pbc):

* **scl**: Predict protein subcellular localization for mixed human and non-human proteins.
* **secondary_structure**: Predict secondary structure of proteins (includes multiple test sets from CASP and the ProtT5
  paper).

### FLIP (Supervised)

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

### PGYM (Zero-Shot)

The `autoeval` module provides support for the
[ProteinGym DMS Supervised](https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/DMS_ProteinGym_substitutions.zip)
datasets that include various *deep mutational scanning* (DMS) fitness scores for protein mutations. For a given model,
a zero-shot method must be selected. Then, it is evaluated on all datasets. The `ProteinGym` datasets were split
into three tasks (using
the [reference file](https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/DMS_substitutions.csv):

* **virus**: Consists of all datasets for which the taxon in the reference file equals `virus`. PLMs are often trained
  without or with only limited viral sequences, so this separate evaluation is interesting to see the effect of data
  distribution shifts after training.
* **non-virus**: All other datasets.
* **total**: **virus** and **non-virus** combined.

> [!IMPORTANT]  
> As ProteinGym does not use a seed for bootstrapping the results, the evaluation results between `autoeval` and
> the ProteinGym leaderboard can differ. Our experiments showed deviations between `+/- 0.01 to 0.025` points for SCC
> on the aggregated results. Please consider this when comparing your results to the ProteinGym leaderboard.

*Please use a different output directory for different zero-shot methods, as the report is currently only able to
capture the results of a single method at once.*

## How to use

### CLI

The `autoeval` pipeline can be run directly through the cli:

```shell
biotrainer autoeval --embedder-name embedder_name --framework framework_name [--min-seq-length min_length] [--max-seq-length max_length] [--use-half-precision]
```

### Script - Supervised

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

### Script - Zero-Shot

Here is a minimal example on how to use the `autoeval_pipeline` for zero-shot evaluation:

```python
from biotrainer.autoeval import autoeval_pipeline
from biotrainer.bioengineer import ZeroShotMethod

print("Starting AutoEval pipeline...")

current_progress = None
for progress in autoeval_pipeline(embedder_name="facebook/esm2_t6_8M_UR50D",
                                  framework="pgym",
                                  zero_shot_method=ZeroShotMethod.MASKED_MARGINALS,
                                  device="cuda",
                                  ):
    print(f"AutoEvalProgress: " + str(progress))
    current_progress = progress

print("**FINISHED**")

current_progress.final_report.summary()
```

## Report

After all tasks have been successfully finished, a report is created in the output directory. All metadata and
model results are tracked there. See the [autoeval_report class](../biotrainer/autoeval/pipelines/autoeval_report.py)
for more details.

The report can be summarized:

```python
final_report = progress.final_report
final_report.summary()
```

and compared to other reports:

```python
final_report.compare([other_reports], plot=True)
```

### Visualization and Leaderboard

**BETA**

You can visualize your results and compare against other embedder
models using [biocentral](https://app.biocentral.cloud). Simply load the report from file in the pLM Evaluation module.
A leaderboard for common protein language models is currently in development.