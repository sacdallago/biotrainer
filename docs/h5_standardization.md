If you need to generate your embeddings `h5` file from your custom embeddings or script, there are two rules you need to follow:

1. embedding dimensions: are you storing a per-sequence or a per-token embedding? I.e. is it a `sequence_to_x` or a `residue[s]_to_x` protocol? Depending on that, your embedding dataset must either be a matrix of `LxD`, or only `D`, with `D` indicating the embedding dimension.
2. you must store the sequence id as an extra attribute (called `original_id`), not as the key of the `h5` dataset, as `h5` keys have special constraints on format which cannot always be satisfied with normal protein identifiers

Here's an example of how to construct an `h5` file, you find more in [examples/custom_embeddings](examples/custom_embeddings):


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
        # Using f"S{i}" to avoid haing integer keys
        output_embeddings_file.create_dataset(f"S{i}", data=protein['embeddings'])

        # !!IMPORTANT:
        # Don't use original sequence id as key in h5 file because h5 keys don't accept special characters
        # re-assign the original id as an attribute to the dataset instead:
        output_embeddings_file[f"S{i}"].attrs["original_id"] = protein['id']
```
