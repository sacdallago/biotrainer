import esm
import h5py
import torch
import logging

from Bio import SeqIO
from tqdm import tqdm
from biotrainer.trainers import CustomEmbedder

logger = logging.getLogger(__name__)


class ESM2Embedder(CustomEmbedder):
    """
    [ESM-2](https://github.com/facebookresearch/esm) large model example for a biotrainer-compatible CustomEmbedder.
    Can be used for per-residue and per-protein embeddings.

    Paper:
    Lin et al. 2022: Language models of protein sequences at the scale of evolution enable accurate structure prediction
    (https://www.biorxiv.org/content/10.1101/2022.07.20.500902v1)
    """

    name: str = "esm2_embedder"

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
        protein_sequences = [
            (record.id, str(record.seq))
            for record in sorted(list(SeqIO.parse(sequence_file, "fasta")), key=lambda record: len(record.seq))
            if len(str(record.seq)) <= 2048
        ]
        logger.info(f"ESM-2: Embedding {len(protein_sequences)} protein sequences!")

        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        batch_converter = alphabet.get_batch_converter()
        model.eval()

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model.to(device)

        batch_labels, batch_strs, batch_tokens = batch_converter(protein_sequences)

        with torch.no_grad():
            embeddings_dict = {}
            for label, seq, tokens in tqdm(zip(batch_labels, batch_strs, batch_tokens), total=len(batch_tokens)):
                embeddings = model(
                    torch.reshape(tokens, (1, tokens.shape[0])).to(device),
                    repr_layers=[33]
                )
                embedding = embeddings["representations"][33][0].detach().cpu()

                if reduce_per_protein:
                    per_protein_embedding = torch.mean(embedding.squeeze(), dim=0)
                    embeddings_dict[label] = per_protein_embedding
                else:
                    embeddings_dict[label] = embedding.squeeze()

        with h5py.File(output_path, "w") as embeddings_file:
            idx = 0
            for seq_id, embedding in embeddings_dict.items():
                embeddings_file.create_dataset(str(idx), data=embedding, compression="gzip", chunks=True)
                embeddings_file[str(idx)].attrs["original_id"] = seq_id  # Follows biotrainer & bio_embeddings standard
                idx += 1

        return output_path
