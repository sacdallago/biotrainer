from biotrainer.inference import Inferencer
from biotrainer.embedders import OneHotEncodingEmbedder


def inference():
    sequences = [
        "PROVTEIN",
        "SEQVENCESEQVENCE"
    ]

    out_file_path = '../residue_to_class/output/out.yml'

    inferencer, out_file = Inferencer.create_from_out_file(out_file_path=out_file_path, allow_torch_pt_loading=True)

    print(f"For the {out_file['model_choice']}, the metrics on the test set are:")
    for metric in out_file['test_iterations_results']['metrics']:
        print(f"\t{metric} : {out_file['test_iterations_results']['metrics'][metric]}")


    embedder = OneHotEncodingEmbedder()
    embeddings = list(embedder.embed_many(sequences))
    # Note that for per-sequence embeddings, you would have to reduce the embeddings now:
    # embeddings = [[embedder.reduce_per_protein(embedding)] for embedding in embeddings]
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