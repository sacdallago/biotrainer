# custom_embedder example

The examples in this folder show how to use a custom embedder that is not included in *bio_embeddings*.

**Example models**:
1. [Ankh](https://github.com/agemagician/Ankh) large model from
Elnaggar et al. 2023: Ankh: Optimized Protein Language Model Unlocks General-Purpose Modelling
(https://doi.org/10.48550/arXiv.2301.06568)

Requires the ankh package:
```bash
pip install ankh
```

2. [ESM-2](https://github.com/facebookresearch/esm) large model from
Lin et al. 2022: Language models of protein sequences at the scale of evolution enable accurate structure prediction
(https://www.biorxiv.org/content/10.1101/2022.07.20.500902v1).

Requires the esm package:
```bash
pip install fair-esm
```

All models need to inherit the following class from `biotrainer.trainers.custom_embedder.py`:
```python
from abc import abstractmethod, ABC


class CustomEmbedder(ABC):
    """
    CustomEmbedder interface that can be inherited to use an embedder that is not included in bio_embeddings.
    Note that some modules (urllib, requests, os) are disabled for the script.
    """

    # Embedder name, also used to create the directory where the computed embeddings are stored
    name: str = "custom_embedder"

    @abstractmethod
    def embed_many(self, sequence_file: str, output_path: str, reduce_per_protein: bool) -> str:
        """
        Abstract method that must be overwritten by a custom embedder.
        It must handle the following steps:
        1. Read the sequences from the provided sequence_file (fasta)
        2. Embed these sequences using the custom embedder
        3. Reduce them to per-protein embeddings if necessary
        4. Write them as h5 File to the given output_path
            -> Make sure to apply the biotrainer/bio_embeddings h5 file standard here:
            Sequence ids must be given as an attribute: embeddings_file[str(idx)].attrs["original_id"] = seq_id

        :param sequence_file: Path to the sequence file
        :param output_path: Output path where to store the generated embeddings
        :param reduce_per_protein: If True, per-residue embeddings must be reduced to per-protein embeddings

        :return: File path of generated embeddings file. Should equal output_path but can be modified if necessary.
        """
        raise NotImplementedError

```