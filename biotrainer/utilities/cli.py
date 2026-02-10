import os
import shutil
import tempfile
import cyclopts

from pathlib import Path
from typing import Union, Dict, Any, Optional, List

from .executer import parse_config_file_and_execute_run

from ..trainers import Pipeline
from ..inference import Inferencer
from ..autoeval import autoeval_pipeline
from ..embedders import get_embedding_service
from ..input_files import convert_deprecated_fastas, read_FASTA, BiotrainerSequenceRecord

app = cyclopts.App()


@app.command
def train(config: Union[str, Path, Dict[str, Any]]) -> Dict[str, Any]:
    """
       Entry point for training

       @param config: Biotrainer configuration file path or config dict
       """
    return parse_config_file_and_execute_run(config)


def train_with_custom_pipeline(config: Union[str, Path, Dict[str, Any]],
                               custom_pipeline: Pipeline) -> Dict[str, Any]:
    """
    Entry point for usage in scripts with a custom pipeline to be executed instead of the default biotrainer pipeline.
    This can, for example, be useful, if embeddings should be loaded from a database instead of a file.

    @param config: Biotrainer configuration file path or config dict
    @param custom_pipeline: custom pipeline to execute
    """
    return parse_config_file_and_execute_run(config, custom_pipeline=custom_pipeline)


def predict_from_records(training_output_file: Union[str, Path],
                         seq_records: List[BiotrainerSequenceRecord],
                         save_embeddings: Optional[bool] = False,
                         scale_embeddings: Optional[bool] = True,
                         ):
    input_ids = {record.get_id_for_id2emb(): record.seq_id for record in seq_records}
    inferencer, iom = Inferencer.create_from_out_file(out_file_path=training_output_file,
                                                      automatic_path_correction=True)

    embedding_service = get_embedding_service(embedder_name=iom.embedder_name(),
                                              custom_tokenizer_config=None,  # TODO
                                              use_half_precision=iom.use_half_precision(),
                                              device=iom.device())
    adapter_path = iom.adapter_path()
    if adapter_path is not None:
        embedding_service.add_finetuned_adapter(adapter_path=adapter_path)

    feature_scaler = iom.feature_scaler() if scale_embeddings else None

    with tempfile.TemporaryDirectory() as tmpdir:
        result_file = embedding_service.compute_embeddings(input_data=seq_records,
                                                           output_dir=Path(tmpdir),
                                                           protocol=iom.protocol(),
                                                           )
        embeddings = embedding_service.load_embeddings(result_file)
        if feature_scaler is not None:
            embeddings = feature_scaler.transform(embeddings)

        if save_embeddings:
            shutil.copy(result_file, os.getcwd())

    result = inferencer.from_embeddings(embeddings=embeddings)["mapped_predictions"]

    sorted_results = []
    for seq_hash, prediction in result.items():
        input_id = input_ids[seq_hash]
        sorted_results.append((input_id, seq_hash, prediction))

    sorted_results = sorted(sorted_results, key=lambda x: x[0])

    for input_id, seq_hash, prediction in sorted_results:
        print(f"Prediction for {input_id} (sequence hash {seq_hash}):\n\t{prediction}")

    return result


@app.command
def predict(training_output_file: Union[str, Path],
            model_input: str,
            save_embeddings: Optional[bool] = False) -> Dict[str, Any]:
    if isinstance(model_input, str):
        if "." in model_input and Path(model_input).exists():
            records = read_FASTA(model_input)
        else:
            model_input_split = [seq for seq in model_input.split(",")]
            records = [BiotrainerSequenceRecord(seq_id=f"Seq{idx}",
                                                seq=seq) for idx, seq in enumerate(model_input_split)]
    else:
        raise ValueError("model_input must be a Path to an input file or a comma separated list of sequences!")

    return predict_from_records(training_output_file=training_output_file,
                               seq_records=records,
                               save_embeddings=save_embeddings)



@app.command
def convert(sequence_file: str,
            labels_file: Optional[str] = None,
            masks_file: Optional[str] = None,
            converted_file: Optional[str] = "converted.fasta",
            skip_inconsistencies: Optional[bool] = False,
            target_format: Optional[str] = "fasta"):
    convert_deprecated_fastas(result_file=converted_file,
                              sequence_file=sequence_file,
                              labels_file=labels_file,
                              masks_file=masks_file,
                              skip_sequence_on_failed_merge=skip_inconsistencies)


@app.command
def autoeval(embedder_name: str,
             framework: str,
             min_seq_length: Optional[int] = 0,
             max_seq_length: Optional[int] = 2000,
             use_half_precision: Optional[bool] = False,
             ):
    for progress in autoeval_pipeline(embedder_name=embedder_name,
                                      framework=framework,
                                      min_seq_length=min_seq_length,
                                      max_seq_length=max_seq_length,
                                      use_half_precision=use_half_precision,
                                      ):
        print(progress)


if __name__ == "__main__":
    app()
