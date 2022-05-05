# Data standardization for biotrainer

In this file, we provide an overview of the data formats we chose for *biotrainer*.
An overview of our decision process and about arguments for and against certain data formats can be found in
`docs/ADR001_data_standardization.md`.

## Table of contents

<!-- toc -->

- [Standardization decisions](#standardization-decisions)
  * [Residue -> Class](#residue---class)
  * [Sequence -> Class](#sequence---class)
  * [Sequence -> Value](#sequence---value)

<!-- tocstop -->

## Standardization decisions

### Residue -> Class

>You have an input protein sequence and want to predict 
for each residue (amino acid) in the sequence a categorical property 
(e.g., residue 4, which is an Alanine, is predicted to be part of an alpha-helix).

**2 Fasta files (sequence.fasta, label.fasta)**

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

### Sequence -> Class

>You have an input protein sequence and want to predict a property for the whole sequence
(e.g. if the sequence is a trans-membrane protein or not).

**1 single Fasta file**
```fasta
# sequences.fasta
>Seq1 TARGET=Glob SET=train VALIDATION=False 
SEQWENCE
```

### Sequence -> Value

>You have an input protein sequence and want to predict the value of a property for the whole sequence
(e.g. the melting temperature of the protein).

**1 single Fasta file**
```fasta
# sequences.fasta
>Seq1 TARGET=37.3452 SET=train VALIDATION=False 
SEQWENCE
```
