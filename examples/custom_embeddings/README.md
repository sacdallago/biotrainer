# Using custom embeddings

This directory contains several notebooks that explain step-by-step, how to add values to an existing h5 file
or even create a new embedding file from existing values. 
The idea behind this is that an experienced, advanced user does not only rely on the provided embedders, but is
able to add any values he or she wants to the embeddings and train a model with them.
The notebooks want to show how this can be achieved, the user is encouraged to apply the existing examples to his
or her use-case.
Please also refer to the [data standardization](../../docs/data_standardization.md#embeddings) 
document to learn about the necessary file standards, that your custom embeddings file has to apply.

All used files for the examples can be found in *example_files*.

## Installation

Make sure you have a virtual environment installed with the *jupyter* extra:
```bash
uv pip install -e ".[jupyter]"
```

## Examples

1. **Add values from csv**:   
Adding *arbitrary_values.csv* to an existing one_hot_encoding embedding on sequence level.   

2. **Concatenate embeddings**:  
Concatenate one_hot_encoding embeddings and word2vec embeddings, both on sequence and residue level.  

3. **Create embeddings from csv**:  
Create a completely new embedding file from *arbitrary_values.csv* on sequence level.  

4. **Nucleotide augmentation**:  
Shows how to augment existing one_hot_encoding embeddings on sequence level with nucleotide ngrams. For more details,
please refer to the [original paper](https://www.biorxiv.org/content/10.1101/2022.03.08.483422v1) 
and [repository](https://github.com/minotm/NTA) by Mason Minot and Sai T. Reddy.
