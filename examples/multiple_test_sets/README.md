# Multiple test sets example

These examples show how to use multiple distinct test sets and the prediction dataset.

## Per-Sequence Tasks

Just add your sequences to the input.fasta file:
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

Similarly, add your targets, masks and sets to the input file:

*Sequence file*:
```fasta
>Seq1 TARGET=DVCDVVDD SET=train
SEQWENCE
>Seq2 TARGET=DDDDDD SET=val
PRTEIN
>Seq3 TARGET=DDDDDDFFEEEDDD SET=test
SEQVENCEPROTEI
>Seq4 TARGET=DDDEEEFFEEEDFF SET=test2
QQQVVVCEPROTEI
>Seq5 TARGET=DDDDDDDDDDD SET=pred # TARGET is ignored during inference
PRTEINNNNNN
```

Execute the example (from the base directory):

```shell
biotrainer train --config examples/multiple_test_sets/sequence_to_class/config.yml
biotrainer train --config examples/multiple_test_sets/residue_to_class/config.yml
```
