# Data standardization for biotrainer

In this file, we provide an overview of the data formats we chose for *biotrainer*.
An overview of our decision process and about arguments for and against certain data formats can be found in
`docs/ADR001_data_standardization.md`.

## Table of contents

<!-- toc -->

- [Before getting started](#before-getting-started)
  * [Sample attributes:](#sample-attributes)
  * [Format](#format)
- [Standardization decisions](#standardization-decisions)
  * [residue_to_class](#residue_to_class)
  * [residues_to_class](#residues_to_class)
  * [sequence_to_class](#sequence_to_class)
  * [sequence_to_value](#sequence_to_value)

<!-- tocstop -->

## Before getting started

### Sample attributes:
  - `SET` --> The partition in which the sample falls. Can be `train` or `test`.
  - `VALIDATION` --> If the sample is used for validation purposes during model training. Can be `True` or `False`.
  - `TARGET` --> The class or value to predict, e.g. `Nucleus` or `1.3`

Mind that `VALIDATION` can only be `True` if the `SET` is `train`, and if `SET` is `test` it must be `False`. 
A combination of `SET=test` and `VALIDATION=True` is a violation.

**All attributes in your .fasta files must exactly match those provided here (case-sensitive!).** 

### Format
All headers of the fasta files expect whitespace separated values, so
```fasta
>Seq1 SET=train VALIDATION=False
DVCDVVDD
```
will work fine, while
```fasta
>Seq1 SET=trainVALIDATION=False
DVCDVVDD
```
or even
```fasta
>Seq1 SET=train;VALIDATION=False
DVCDVVDD
```
will fail.

## Standardization decisions

### residue_to_class
```text
Predict a class C for each residue encoded in D dimensions in a sequence of length L. Input BxLxD --> output BxLxC
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
>Seq1 SET=train VALIDATION=False
DVCDVVDD
```

### residues_to_class
```text
Predict a class C for all residues encoded in D dimensions in a sequence of length L. Input BxLxD --> output BxC
```

You have an input protein sequence and want to predict a property for the whole sequence 
(e.g. the sub-cellular-location of the protein), but you want to use *per-residue embeddings* for the task.

**Required file: FASTA file containing sequences and labels**

sequences.fasta
```fasta
>Seq1 TARGET=Nucleus SET=train VALIDATION=False 
SEQWENCE
```

### sequence_to_class
```text
Predict a class C for each sequence encoded in a fixed dimension D. Input BxD --> output BxC
```

You have an input protein sequence and want to predict a property for the whole sequence
(e.g. if the sequence is a trans-membrane protein or not).

**Required file: FASTA file containing sequences and labels**

sequences.fasta
```fasta
>Seq1 TARGET=Glob SET=train VALIDATION=False 
SEQWENCE
```

### sequence_to_value
```text
Predict a value V for each sequence encoded in a fixed dimension D. Input BxD --> output BxV
```

You have an input protein sequence and want to predict the value of a property for the whole sequence
(e.g. the melting temperature of the protein).

**Required file: FASTA file containing sequences and labels**

sequences.fasta
```fasta
>Seq1 TARGET=37.3452 SET=train VALIDATION=False 
SEQWENCE
```
