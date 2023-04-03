import ankh
import h5py
import torch
import logging

from Bio import SeqIO
from tqdm import tqdm
from biotrainer.trainers import CustomEmbedder

logger = logging.getLogger(__name__)


class AnkhEmbedder(CustomEmbedder):
    """
    [Ankh](https://github.com/agemagician/Ankh) large model example for a biotrainer-compatible CustomEmbedder.
    Can be used for per-residue and per-protein embeddings.

    Paper:
    Elnaggar et al. 2023: Ankh: Optimized Protein Language Model Unlocks General-Purpose Modelling
    (https://doi.org/10.48550/arXiv.2301.06568)
    """

    name: str = "ankh_embedder"

    def embed_many(self, sequence_file: str, output_path: str, reduce_per_protein: bool) -> str:
        """
        Method to embed all sequences from the provided sequence file.

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
        protein_sequences = {seq.id: list(seq.seq) for seq in sorted(list(SeqIO.parse(sequence_file, "fasta")),
                                                                     key=lambda seq: len(seq.seq),
                                                                     reverse=True)}
        logger.info(f"Ankh: Embedding {len(protein_sequences)} protein sequences!")

        model, tokenizer = ankh.load_large_model()
        model.eval()

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model.to(device)

        with torch.no_grad():
            embeddings_dict = {}
            for seq_id, sequence in tqdm(protein_sequences.items()):
                outputs = tokenizer.batch_encode_plus([sequence],
                                                      add_special_tokens=False,
                                                      padding=True,
                                                      is_split_into_words=True,
                                                      return_tensors="pt")
                embeddings = model(input_ids=outputs['input_ids'].to(device),
                                   attention_mask=outputs['attention_mask'].to(device))
                embedding = embeddings[0].detach().cpu()
                if reduce_per_protein:
                    per_protein_embedding = torch.mean(embedding.squeeze(), dim=0)
                    embeddings_dict[seq_id] = per_protein_embedding
                else:
                    embeddings_dict[seq_id] = embedding.squeeze()

        with h5py.File(output_path, "w") as embeddings_file:
            idx = 0
            for seq_id, embedding in embeddings_dict.items():
                embeddings_file.create_dataset(str(idx), data=embedding, compression="gzip", chunks=True)
                embeddings_file[str(idx)].attrs["original_id"] = seq_id  # Follows biotrainer & bio_embeddings standard
                idx += 1
        """
        # Verify h5 file - This is not a necessary step. Only verify your file if you want to be sure that the 
        # computed embeddings are correct. 
        # An example of how this could look like is given here for per-protein embeddings.
        
        if reduce_per_protein:
            ankh_large_dim = 1536
            created_file = h5py.File(output_file_path, 'r')
            for idx, embedding in created_file.items():
                original_sequence_id = created_file[idx].attrs["original_id"]
                assert embedding.shape[0] == ankh_large_dim, "New dimension is not correct"
                assert embedding[0] == embeddings_dict[original_sequence_id][0]
            assert len(created_file.keys()) == len(protein_sequences.keys())
            created_file.close()
        """
        return output_path
