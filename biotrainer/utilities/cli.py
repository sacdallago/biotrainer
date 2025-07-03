import os
import shutil
import tempfile
import cyclopts

from pathlib import Path
from typing import Union, Dict, Any, Optional

from .hashing import calculate_sequence_hash
from .executer import parse_config_file_and_execute_run

from ..trainers import Pipeline
from ..inference import Inferencer
from ..autoeval import autoeval_pipeline
from ..embedders import get_embedding_service
from ..input_files import convert_deprecated_fastas, read_FASTA

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


@app.command
def predict(training_output_file: Union[str, Path], model_input: str,
            save_embeddings: Optional[bool] = False) -> Dict[str, Any]:
    if isinstance(model_input, str):
        if "." in model_input and Path(model_input).exists():
            model_input = read_FASTA(model_input)
            input_ids = {seq_record.get_hash(): seq_record.seq_id for seq_record in model_input}
            model_input = {seq_record.get_hash(): seq_record.seq for seq_record in model_input}
        else:
            model_input = [seq for seq in model_input.split(",")]
            input_ids = {calculate_sequence_hash(seq): f"Seq{idx}" for idx, seq in enumerate(model_input)}
    else:
        raise ValueError("model_input must be a Path to an input file or a comma separated list of sequences!")

    inferencer, iom = Inferencer.create_from_out_file(out_file_path=training_output_file,
                                                      automatic_path_correction=True)

    embedding_service = get_embedding_service(embedder_name=iom.embedder_name(),
                                              custom_tokenizer_config=None,  # TODO
                                              use_half_precision=iom.use_half_precision(),
                                              device=iom.device())
    adapter_path = iom.adapter_path()
    if adapter_path is not None:
        embedding_service.add_finetuned_adapter(adapter_path=adapter_path)

    with tempfile.TemporaryDirectory() as tmpdir:
        result_file = embedding_service.compute_embeddings(input_data=model_input,
                                                           output_dir=Path(tmpdir),
                                                           protocol=iom.protocol(),
                                                           )
        embeddings = embedding_service.load_embeddings(result_file)
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
