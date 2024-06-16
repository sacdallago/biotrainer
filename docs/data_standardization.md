# Data standardization for biotrainer

In this file, we provide an overview of the data formats we chose for *biotrainer*.
An overview of our decision process and about arguments for and against certain data formats can be found in
`docs/ADR001_data_standardization.md`.

## Table of contents

<!-- toc -->

- [Before getting started](#before-getting-started)
  * [Sample attributes](#sample-attributes)
  * [Format](#format)
- [Standardizations](#standardizations)
  * [Embeddings](#embeddings)
  * [Overview about the protocols](#overview-about-the-protocols)
  * [residue_to_class](#residue_to_class)
  * [residues_to_class](#residues_to_class)
  * [sequence_to_class](#sequence_to_class)
  * [sequence_to_value](#sequence_to_value)
  * [protein_protein_interaction](#protein_protein_interaction)

<!-- tocstop -->

## Before getting started

### Sample attributes
  - `ID` --> Identifier of the sequence, right after ">" (.fasta standard) 
  - `SET` --> The partition in which the sample falls. Can be `train`, `val` or `test`.
  - `TARGET` --> The class or value to predict, e.g. `Nucleus` or `1.3`

<details>
<summary>
Deprecated VALIDATION annotation
</summary>
Validation annotation can also be given via a separate attribute "VALIDATION=True/False". This behaviour
id deprecated and mutually exclusive with annotations that include a SET=val value.

- `VALIDATION` --> If the sample is used for validation purposes during model training. Can be `True` or `False`.
  
Mind that `VALIDATION` can only be `True` if the `SET` is `train`, and if `SET` is `test` it must be `False`. 
A combination of `SET=test` and `VALIDATION=True` is a violation.
A combination of `SET=val` and `VALIDATION=True` or `VALIDATION=False` is also a violation.
</details>

**All attributes in your .fasta files must exactly match those provided here (case-sensitive!).** 

### Format
All headers of the fasta files expect whitespace separated values, so
```fasta
>Seq1 SET=train TARGET=0.1
SEQWENCE
```
will work fine, while
```fasta
>Seq1 SET=trainTARGET=0.1
SEQWENCE
```
or even
```fasta
>Seq1 SET=train;TARGET=0.1
SEQWENCE
```
will fail.

## Standardizations

### Embeddings

All embeddings used as model input must follow the same standardization requirements. 
* **Format**: Embeddings are stored in [hierarchical data format](https://en.wikipedia.org/wiki/Hierarchical_Data_Format),
usually using the file ending *.h5*.
* **Sequence ID mapping**: Given a fasta file with sequence ids [Seq1, Seq2, ..., SeqN], the corresponding embedding
is stored in a dataset in the h5-file that has the attribute "original_id" set to the associated sequence id:
```python
embeddings_file[idx].attrs["original_id"] = sequence_id
```
This is necessary, because datasets in .h5 files are stored with identifiers (idx) "0" - "N", which does not allow
for a direct mapping from sequence id to embedding.
* **Dimensions**: The final dimensions of the embedding file differ between per-residue and per-sequence embeddings:
```text
# Per-sequence embeddings
Number_Sequences x Embeddings_Dimension
# Per-residue embeddings
Number_Sequences x Sequence_Length x Embeddings_Dimension
```
Note that the *embeddings dimension* must be equal for all residues or sequences, but the *sequence length* can differ.

Some embedders use padded per-residue embeddings that always have the same size regardless of the sequence length. 
Some also include a "start" or "stop" token, so that the dimension of the embedding does not exactly match the 
number of residues. 
**For biotrainer to work with per-residue embeddings, there must be an exact 1:1 match 
between number of residues and embeddings!**

Here's an example of how to construct an `h5` file for a "per-sequence" dataset, you find more examples in [examples/custom_embeddings](examples/custom_embeddings):


```
import h5py

per_sequence_embeddings_path = "/path/to/disk/file.h5"

proteins = [
  {
    'id': "My fav sequence",
    'embeddings': [4,3,2,4]
    'sequence': 'SEQVENCE'
  }
]

with h5py.File(per_sequence_embeddings_path, "w") as output_embeddings_file:
    for i, protein in enumerate(proteins):
        # Using f"S{i}" to avoid having integer keys
        output_embeddings_file.create_dataset(f"S{i}", data=protein['embeddings'])

        # !!IMPORTANT:
        # Don't use original sequence id as key in h5 file because h5 keys don't accept special characters
        # re-assign the original id as an attribute to the dataset instead:
        output_embeddings_file[f"S{i}"].attrs["original_id"] = protein['id']
```

### Overview about the protocols

```text
D=embedding dimension (e.g. 1024)
B=batch dimension (e.g. 30)
L=sequence dimension (e.g. 350)
C=number of classes (e.g. 13)

- residue_to_class --> Predict a class C for each residue encoded in D dimensions in a sequence of length L. Input BxLxD --> output BxLx1
- residues_to_class --> Predict a class C for all residues encoded in D dimensions in a sequence of length L. Input BxLxD --> output Bx1
- sequence_to_class --> Predict a class C for each sequence encoded in a fixed dimension D. Input BxD --> output Bx1
- sequence_to_value --> Predict a value V for each sequence encoded in a fixed dimension D. Input BxD --> output Bx1
```

### residue_to_class
```text
Predict a class C for each residue encoded in D dimensions in a sequence of length L. Input BxLxD --> output BxLx1
```

You have an input protein sequence and want to predict 
for each residue (amino acid) in the sequence a categorical property 
(e.g., residue 4, which is an Alanine, is predicted to be part of an alpha-helix).

**Required files: 2 Fasta files (sequence.fasta, label.fasta)**

sequences.fasta
```fasta
>Seq1
SEQWENCE
```

labels.fasta
```fasta
>Seq1 SET=train
DVCDVVDD
```

### residues_to_class
```text
Predict a class C for all residues encoded in D dimensions in a sequence of length L. Input BxLxD --> output Bx1
```

You have an input protein sequence and want to predict a property for the whole sequence 
(e.g. the sub-cellular-location of the protein), but you want to use *per-residue embeddings* for the task.

**Required file: FASTA file containing sequences and labels**

sequences.fasta
```fasta
>Seq1 TARGET=Nucleus SET=train
SEQWENCE
```

### residues_to_value
```text
Predict a value V for all residues encoded in D dimensions in a sequence of length L. Input BxLxD --> output Bx1
```

You have an input protein sequence and want to predict a property for the whole sequence 
(e.g. the meltdown temperature of the protein), but you want to use *per-residue embeddings* for the task.

**Required file: FASTA file containing sequences and labels**

sequences.fasta
```fasta
>Seq1 TARGET=42.09 SET=train
SEQWENCE
```

### sequence_to_class
```text
Predict a class C for each sequence encoded in a fixed dimension D. Input BxD --> output Bx1
```

You have an input protein sequence and want to predict a property for the whole sequence
(e.g. if the sequence is a trans-membrane protein or not).

**Required file: FASTA file containing sequences and labels**

sequences.fasta
```fasta
>Seq1 TARGET=Glob SET=train
SEQWENCE
```

### sequence_to_value
```text
Predict a value V for each sequence encoded in a fixed dimension D. Input BxD --> output Bx1
```

You have an input protein sequence and want to predict the value of a property for the whole sequence
(e.g. the melting temperature of the protein).

**Required file: FASTA file containing sequences and labels**

sequences.fasta
```fasta
>Seq1 TARGET=37.3452 SET=train
SEQWENCE
```

### protein_protein_interaction

This is not a protocol in and out of itself, but can be applied to any protocol by setting the config option
`interaction`:
```yaml
# config.yml
protocol: sequence_to_class
interaction: multiply | concat  # Default: None
```

So, you have two input proteins and want to predict, if they interact or not (per-sequence interaction prediction).
Hence, the labels and outputs will be in `[0, 1]` (binary classification task).
Before the training, protein embeddings can be computed as usual via the `embedder_name` option.

**Required file: FASTA file containing interactions and labels**

interactions.fasta
```fasta
>Seq1 INTERACTOR=Seq2 TARGET=0 SET=train
SEQWENCE
>Seq2 INTERACTOR=Seq1 TARGET=0 SET=train
PRTEIN
```

Note that each pair of `>ID` and associated sequence must be present *at least once* in the file! Hence, duplicating
each interaction is not strictly necessary.
So, this is also a correct input file:
```fasta
>Seq1 INTERACTOR=Seq2 TARGET=0 SET=train
SEQWENCE
>Seq2 INTERACTOR=Seq1 TARGET=0 SET=train
PRTEIN
>Seq3 INTERACTOR=Seq2 TARGET=1 SET=test
SEQPRTEIN
```
