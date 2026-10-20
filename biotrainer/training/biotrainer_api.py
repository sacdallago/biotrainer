import os
import shutil
import tempfile

from pathlib import Path
from junban import Pipeline
from typing import Union, Dict, Any, Optional, List
from biotrainer_core.input_files import read_FASTA
from biotrainer_core.data_classes import BiotrainerModelResult, SequenceData, BiotrainerInferenceResult

from .inference import Inferencer
from .trainers.pipeline_context import BiotrainerPipelineContext
from .utilities.executer import parse_config_file_and_execute_run
from .output_files import BiotrainerOutputObserver, InferenceOutputManager

from ..embedding import get_embedding_service


class BiotrainerModel:
    def __init__(self, training_result: Optional[BiotrainerModelResult] = None, output_file_path: Optional[Path] = None):
        self.training_result = training_result
        self._output_file_path = output_file_path
        self._inferencer: Optional[Inferencer] = None
        self._iom: Optional[InferenceOutputManager] = None

    @classmethod
    def from_training_result(cls, training_result: Union[BiotrainerModelResult, Path, str]):
        if isinstance(training_result, BiotrainerModelResult):
            return cls(training_result=training_result)
        elif isinstance(training_result, (Path, str)):
            return cls(training_result=BiotrainerModelResult.from_file(training_result),
                       output_file_path=Path(training_result))
        else:
            raise TypeError(f"Invalid type for training_result: {type(training_result)}")

    def train(self,
              config: Union[str, Path, Dict[str, Any]],
              input_data: Optional[List[SequenceData]] = None,
              custom_pipeline: Optional[Pipeline[BiotrainerPipelineContext]] = None,
              custom_output_observers: Optional[List[BiotrainerOutputObserver]] = None,
              write_to_file: Optional[bool] = True) -> BiotrainerModelResult:
        """
        Train a model using the provided configuration and optional custom pipeline and output observers.

        :param config: Biotrainer configuration file path or dictionary.
        :param input_data: Optional list of SequenceData objects for training. If None, data will be loaded from config.
        :param custom_pipeline: A custom pipeline for the training process.
        :param custom_output_observers: Custom Observers for the output (e.g. for tensorboard)
        :param write_to_file: If True, the training result will be written to a file (out.yml).
        :return: BiotrainerModelResult object containing the trained model results.
        """
        if self.training_result is not None:
            print(f"Warning: Training result already available! Overwriting with new training result..")

        training_result = parse_config_file_and_execute_run(config=config,
                                                            input_data=input_data,
                                                            custom_pipeline=custom_pipeline,
                                                            custom_output_observers=custom_output_observers,
                                                            write_to_file=write_to_file)
        self.training_result = training_result
        return training_result

    def inferencer(self) -> Inferencer:
        if self._inferencer is not None:
            return self._inferencer
        if self.training_result is None:
            raise ValueError("No training result available!")
        inf, iom = Inferencer.from_training_result(training_result=self.training_result,
                                                   out_file_path=self._output_file_path,
                                                   automatic_path_correction=True)
        self._inferencer = inf
        self._iom = iom
        return inf

    def inference_output_manager(self) -> InferenceOutputManager:
        if self._iom is not None:
            return self._iom
        _ = self.inferencer()
        return self._iom

    def _predict_from_records(self,
                              seq_data: List[SequenceData],
                              save_embeddings: Optional[bool] = False,
                              scale_embeddings: Optional[bool] = True,
                              ) -> BiotrainerInferenceResult:
        input_ids = {record.get_id_for_id2emb(): record.seq_id for record in seq_data}
        inf = self.inferencer()
        iom = self.inference_output_manager()

        embedding_service = get_embedding_service(
            embedder_name=iom.embedder_name(),
            custom_tokenizer_config=None,  # TODO
            use_half_precision=iom.use_half_precision(),
            device=iom.device(),
            seed=iom.seed(),
        )
        adapter_path = iom.adapter_path()
        if adapter_path is not None:
            embedding_service.add_finetuned_adapter(adapter_path=adapter_path)

        with tempfile.TemporaryDirectory() as tmpdir:
            result_file = embedding_service.compute_embeddings(input_data=seq_data,
                                                               output_dir=Path(tmpdir),
                                                               protocol=iom.protocol(),
                                                               )
            embeddings = embedding_service.load_embeddings(result_file)

            if save_embeddings:
                shutil.copy(result_file, os.getcwd())

        inference_result = inf.from_embeddings(embeddings=embeddings, scale_embeddings=scale_embeddings)

        sorted_results = []
        for prediction in inference_result.predictions:
            input_id = input_ids[prediction.seq_id]
            sorted_results.append((input_id, prediction.seq_id, prediction))

        sorted_results = sorted(sorted_results, key=lambda x: x[0])

        for input_id, seq_hash, prediction in sorted_results:
            print(f"Prediction for {input_id} (sequence hash {seq_hash}):\n\t{prediction}")

        return inference_result

    def predict(self,
                model_input: Union[str, List[SequenceData]],
                save_embeddings: Optional[bool] = False,
                scale_embeddings: Optional[bool] = True,
                ) -> BiotrainerInferenceResult:
        """ Convenience function to create predictions from a fasta file or sequence data list.
            Automatically embeds the sequences (as defined in the training config) and calculates predictions.
            Use the .inferencer() method to get a more flexible and powerful interface.
        """
        if self.training_result is None:
            raise ValueError("No training result available!")

        if self.training_result.config["model_choice"] == "GP":
            raise NotImplementedError("GP inference not yet implemented.")

        if isinstance(model_input, list) and isinstance(model_input[0], SequenceData):
            sequence_data = model_input
        elif isinstance(model_input, str):
            if "." in model_input and Path(model_input).exists():
                sequence_data = read_FASTA(model_input)
            else:
                model_input_split = [seq for seq in model_input.split(",")]
                sequence_data = [SequenceData(seq_id=f"Seq{idx}",
                                              seq=seq) for idx, seq in enumerate(model_input_split)]
        else:
            raise ValueError("model_input must be a Path to an input file or a comma separated list of sequences!")

        return self._predict_from_records(
            seq_data=sequence_data,
            save_embeddings=save_embeddings,
            scale_embeddings=scale_embeddings
        )
