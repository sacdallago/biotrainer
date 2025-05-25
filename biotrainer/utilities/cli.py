import os
import shutil
import tempfile
import cyclopts

from pathlib import Path
from typing import Union, Dict, Any, Callable, List, Optional

from .executer import parse_config_file_and_execute_run

from ..inference import Inferencer
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


def train_with_custom_trainer(config: Union[str, Path, Dict[str, Any]],
                              custom_trainer_function: Callable) -> Dict[str, Any]:
    """
    Entry point for usage in scripts with a function to create a custom trainer.
    The custom trainer function must take the trainer parameters as input and return a (subclass of) Trainer e.g.
    custom_trainer_function = lambda hp_manager, output_vars, config: Trainer(hp_manager=hp_manager,
                                                                              output_vars=output_vars,
                                                                              **config
                                                                              )

    @param config: Biotrainer configuration file path or config dict
    """
    return parse_config_file_and_execute_run(config, custom_trainer_function=custom_trainer_function)


@app.command
def predict(training_output_file: Union[str, Path], model_input: str,
            save_embeddings: Optional[bool] = False) -> Dict[str, Any]:
    if isinstance(model_input, str):
        if "." in model_input and Path(model_input).exists():
            model_input = read_FASTA(model_input)
            model_input = {seq_record.seq_id: seq_record.seq for seq_record in model_input.values()}
        else:
            model_input = {f"Seq{idx}": seq for idx, seq in enumerate(model_input.split(","))}
    else:
        raise ValueError("model_input must be a Path to an input file or a comma separated list of sequences!")

    inferencer, iom = Inferencer.create_from_out_file(out_file_path=training_output_file,
                                                         automatic_path_correction=True)

    embedding_service = get_embedding_service(embeddings_file_path=None,
                                              embedder_name=iom.embedder_name(),
                                              custom_tokenizer_config=None,  # TODO
                                              use_half_precision=iom.use_half_precision(),
                                              device=iom.device())

    with tempfile.TemporaryDirectory() as tmpdir:
        result_file = embedding_service.compute_embeddings(input_data=model_input,
                                                           output_dir=Path(tmpdir),
                                                           protocol=iom.protocol(),
                                                           )
        embeddings = embedding_service.load_embeddings(result_file)
        if save_embeddings:
            shutil.copy(result_file, os.getcwd())

    result = inferencer.from_embeddings(embeddings=embeddings)["mapped_predictions"]

    print(result)
    return result


@app.command
def convert(sequence_file: str,
            labels_file: Optional[str] = None,
            masks_file: Optional[str] = None,
            converted_file: Optional[str] = "converted.fasta",
            target_format: Optional[str] = "fasta"):
    convert_deprecated_fastas(result_file=converted_file,
                              sequence_file=sequence_file,
                              labels_file=labels_file,
                              masks_file=masks_file)


if __name__ == "__main__":
    app()
