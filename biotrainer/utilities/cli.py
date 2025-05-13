import cyclopts

from pathlib import Path
from typing import Union, Dict, Any, Callable, List, Optional

from .executer import parse_config_file_and_execute_run

from ..input_files import convert_deprecated_fastas


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
def inference(training_output_file: Union[str, Path], model_input: Union[str, List[str], Path]) -> int:
    # TODO
    return 0

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
