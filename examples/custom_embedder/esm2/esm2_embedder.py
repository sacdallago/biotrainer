import esm
import torch
import logging

from tqdm import tqdm
from numpy import ndarray
from biotrainer.embedders import CustomEmbedder
from typing import Generator, Iterable, Optional

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

    def embed_many(
            self, sequences: Iterable[str], batch_size: Optional[int] = None
    ) -> Generator[ndarray, None, None]:
        """
        Method to embed all sequences from the provided iterable.
        This is the function that should be overwritten by most custom embedders, because it allows full control
        over the whole embeddings generation process. Other functions are optional to use and overwrite, except
        reduce_per_protein (if necessary).

        Yields embedding for one sequence at a time.

        :param sequences: List of proteins as AA strings
        :param batch_size: For embedders that profit from batching, this is maximum number of AA per batch

        :return: A list object with embeddings of the sequences.
        """

        logger.info(f"ESM-2: Embedding protein sequences!")

        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        batch_converter = alphabet.get_batch_converter()
        model.eval()

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model.to(device)

        batch_labels, batch_strs, batch_tokens = batch_converter(sequences)

        with torch.no_grad():
            for label, seq, tokens in tqdm(zip(batch_labels, batch_strs, batch_tokens), total=len(batch_tokens)):
                embeddings = model(
                    torch.reshape(tokens, (1, tokens.shape[0])).to(device),
                    repr_layers=[33]
                )
                yield embeddings["representations"][33][0].detach().cpu()

    @staticmethod
    def reduce_per_protein(embedding: ndarray) -> ndarray:
        return torch.mean(torch.tensor(embedding).squeeze(), dim=0).numpy()
