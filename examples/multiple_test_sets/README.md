# Multiple test sets example

These examples show how to use multiple distinct test sets and the prediction dataset.

## Per-Sequence Tasks

Just add your sequences to the sequence.fasta file:
```fasta
>Seq1 TARGET=Glob SET=train
SEQWENCE
>Seq2 TARGET=GlobSP SET=val
PRTEIN
>Seq3 TARGET=TM SET=test
SEQVENCEPROTEI
>Seq4 TARGET=TMSP SET=test
PRTEINSEQWENCE
>Seq5 TARGET=TM SET=test2
PRTEIIIIINSEQWENCE
>Seq6 TARGET=Glob SET=test3
PRTSEQ
>Seq7 SET=pred
PRRRRTSEQQQ
```

## Per-Residue Tasks

You need to add your sequences and labels (and masks) to the respective files, set annotations must be done in the
labels fasta file. For the prediction set, you currently need to add a label to the value as well (although it will
be ignored of course during inference):

*Sequence file*:
```fasta
>Seq1
SEQWENCE
>Seq2
PRTEIN
>Seq3
SEQVENCEPROTEI
>Seq4
QQQVVVCEPROTEI
>Seq5
PRTEINNNNNN
```

*Labels file*:
```fasta
>Seq1 SET=train
DVCDVVDD
>Seq2 SET=val
DDDDDD
>Seq3 Inhibitor of nuclear factor kappa-B kinase subunit beta OS=Homo sapiens OX=9606 GN=IKBKB PE=1 SV=1 SET=test
DDDDDDFFEEEDDD
>Seq4 SET=test2
DDDEEEFFEEEDFF
>Seq5 SET=pred
DDDEEEFFEEE  # This is ignored during inference
```

This behaviour will be improved in a future release of biotrainer.

Execute the example (from the base directory):

```shell
poetry run python3 run-biotrainer.py examples/multiple_test_sets/sequence_to_class/config.yml
poetry run python3 run-biotrainer.py examples/multiple_test_sets/residue_to_class/config.yml
```
