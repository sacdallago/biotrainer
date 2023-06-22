import os

from ruamel import yaml
from pathlib import Path
from ruamel.yaml import YAMLError
from typing import Union, List, Dict, Any

from ..protocols import Protocol
from .config_option import ConfigurationException, ConfigOption, FileOption
from .config_rules import MutualExclusive, ProtocolRequires
from .input_options import SequenceFile, LabelsFile, input_options
from .training_options import AutoResume, PretrainedModel, training_options
from .embedding_options import EmbedderName, EmbeddingsFile, embedding_options
from .general_options import general_options
from .model_options import model_options

protocol_rules = [
    ProtocolRequires(protocol=Protocol.per_residue_protocols(), requires=[SequenceFile, LabelsFile]),
    ProtocolRequires(protocol=Protocol.per_sequence_protocols(), requires=[SequenceFile]),
]

config_option_rules = [
    MutualExclusive(exclusive=[AutoResume, PretrainedModel], error_message="Use auto_resume in case you need to "
                                                                           "restart your training job multiple times.\n"
                                                                           "Use pretrained_model if you want to "
                                                                           "continue to train a specific model."),
    MutualExclusive(exclusive=[EmbedderName, EmbeddingsFile],
                    allowed_values=["custom_embeddings"],
                    error_message="Please provide either an embedder_name to calculate embeddings from scratch or \n"
                                  "an embeddings_file to use pre-computed embeddings.")
]

all_options_dict: Dict[str, ConfigOption] = {option.name: option for option in
                                             general_options + input_options + model_options +
                                             training_options + embedding_options}


class Configurator:

    def __init__(self, config_dict: Dict, config_file_path: Path = None):
        if not config_file_path:
            config_file_path = Path("")
        self._config_file_path = config_file_path
        self._config_dict = config_dict
        self.protocol = self._get_protocol_from_config_dict(config_dict)

    @classmethod
    def from_config_dict(cls, config_dict: Dict[str, Any]):
        return cls(config_dict=config_dict)

    @classmethod
    def from_config_path(cls, config_path: Union[str, Path]):
        return cls(config_dict=cls._read_config_file(config_path),
                   config_file_path=Path(os.path.dirname(os.path.abspath(config_path))))

    @staticmethod
    def get_option_dicts_by_protocol(protocol: Protocol):
        result = []
        for option_class in all_options_dict.values():
            option = option_class(protocol=protocol, value="")
            if protocol in option.allowed_protocols:
                result.append(option.to_dict())
        return result

    @staticmethod
    def _read_config_file(config_path: Union[str, Path], preserve_order: bool = True) -> dict:
        """
        Read config from path to file.

        :param config_path: path to .yml config file
        :param preserve_order: Preserve order in file
        :return: Config file parsed as dict
        """
        with open(config_path, "r") as fp:
            try:
                if preserve_order:
                    return yaml.load(fp, Loader=yaml.Loader)
                else:
                    return yaml.safe_load(fp)
            except YAMLError as e:
                raise ConfigurationException(
                    f"Could not parse configuration file at '{config_path}' as yaml. "
                    "Formatting mistake in config file? "
                    "See Error above for details."
                ) from e

    @staticmethod
    def _get_protocol_from_config_dict(config_dict: Dict[str, Any]):
        try:
            protocol = config_dict["protocol"]
            return Protocol[protocol]
        except KeyError:
            raise ConfigurationException(f"No protocol specified in config file!")

    @staticmethod
    def _get_config_map(protocol: Protocol, config_dict: Dict[str, Any], config_file_path: Path = None) \
            -> Dict[str, ConfigOption]:

        config_map = {}
        for config_name in config_dict.keys():
            try:
                value = config_dict[config_name]
                config_object: ConfigOption = all_options_dict[config_name](protocol=protocol, value=value)
                # Download file if necessary and allowed
                if issubclass(config_object.__class__, FileOption):
                    config_object.download_file_if_necessary(value, config_file_path)
                    if config_file_path:
                        config_object.make_file_path_absolute(config_file_path)
                config_map[config_name] = config_object
            except KeyError:
                raise ConfigurationException(f"Unknown configuration option: {config_name}!")
        # Add default values
        all_options_for_protocol: List[ConfigOption] = [option for option in
                                                        all_options_dict.values()
                                                        if protocol in option.allowed_protocols]
        for option in all_options_for_protocol:
            config_object = option(protocol=protocol)
            if config_object.name not in config_dict.keys() and config_object.default_value != "":
                if issubclass(config_object.__class__, FileOption):
                    config_object.make_file_path_absolute(config_file_path)
                config_map[option.name] = config_object

        return config_map

    @staticmethod
    def _verify_config(protocol: Protocol, config_map: Dict[str, ConfigOption]) -> bool:
        # Check rules
        config_objects = list([config_object for config_object in config_map.values()])
        all_rules = protocol_rules + config_option_rules
        for rule in all_rules:
            success, reason = rule.apply(protocol=protocol,
                                         config=config_objects)
            if not success:
                raise ConfigurationException(reason)

        for config_object in config_objects:
            # Check protocol
            if protocol not in config_object.allowed_protocols:
                raise ConfigurationException(f"{config_object.name} not allowed for protocol {protocol}.")
            # Check value
            if not config_object.is_value_valid():
                raise ConfigurationException(f"{config_object.value} not valid for option {config_object.name}.")

        return True

    def get_verified_config(self) -> Dict[str, Any]:
        config_map: Dict[str, ConfigOption] = self._get_config_map(protocol=self.protocol,
                                                                    config_dict=self._config_dict,
                                                                    config_file_path=self._config_file_path)
        self._verify_config(protocol=self.protocol, config_map=config_map)
        result = {}
        for config_object in config_map.values():
            result[config_object.name] = config_object.value

        return result
