from biotrainer.inference import Inferencer
from biotrainer.embedders import OneHotEncodingEmbedder


def inference():
    sequences = [
        "PROVTEIN",
        "SEQVENCESEQVENCE"
    ]

    out_file_path = '../residue_to_class/output/out.yml'

    inferencer, iom = Inferencer.create_from_out_file(out_file_path=out_file_path)

    embedder = OneHotEncodingEmbedder()
    embeddings = list(embedder.embed_many(sequences))
    # Note that for per-sequence embeddings, you would have to reduce the embeddings now:
    # embeddings = [embedder.reduce_per_protein(embedding) for embedding in embeddings]
    predictions = inferencer.from_embeddings(embeddings, split_name="hold_out")
    for sequence, prediction in zip(sequences, predictions["mapped_predictions"].values()):
        print(sequence)
        print(prediction)

    # If your checkpoints are stored as .pt, consider converting them to safetensors (supported by biotrainer >=0.9.1)
    inferencer.convert_all_checkpoints_to_safetensors()


def main(args=None):
    inference()


if __name__ == '__main__':
    main()
