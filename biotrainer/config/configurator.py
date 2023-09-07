import os

from ruamel import yaml
from pathlib import Path
from ruamel.yaml import YAMLError
from typing import Union, List, Dict, Any, Tuple

from .model_options import model_options
from .general_options import general_options
from .input_options import SequenceFile, LabelsFile, input_options
from .config_option import ConfigurationException, ConfigOption
from .training_options import AutoResume, PretrainedModel, training_options
from .embedding_options import EmbedderName, EmbeddingsFile, embedding_options
from .config_rules import (MutualExclusive, ProtocolRequires, OptionValueRequires,
                           AllowHyperparameterOptimization)
from .cross_validation_options import (cross_validation_options, CROSS_VALIDATION_CONFIG_KEY, Method, ChooseBy,
                                       CrossValidationOption, K, Nested, NestedK, SearchMethod, NMaxEvaluationsRandom,
                                       P)

from ..protocols import Protocol

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
                                  "an embeddings_file to use pre-computed embeddings."),
]

optimization_rules = [
    AllowHyperparameterOptimization(option=Nested, value=True),
]

cross_validation_rules = [
    OptionValueRequires(option=Method, value="k_fold", requires=[K]),
    OptionValueRequires(option=Nested, value=True, requires=[NestedK, SearchMethod]),
    OptionValueRequires(option=SearchMethod, value="random_search", requires=[NMaxEvaluationsRandom]),
    OptionValueRequires(option=Method, value="leave_p_out", requires=[P]),
]

all_options_dict: Dict[str, ConfigOption] = {option.name: option for option in
                                             general_options + input_options + model_options +
                                             training_options + embedding_options}

cross_validation_dict: Dict[str, ConfigOption] = {option.name: option for option in cross_validation_options}


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
    def get_option_dicts_by_protocol(protocol: Protocol, include_cross_validation_options: bool = False):
        result = []
        all_config_options_dict = all_options_dict | cross_validation_dict \
            if include_cross_validation_options else all_options_dict
        for option_class in all_config_options_dict.values():
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
    def _get_cross_validation_map(protocol: Protocol, cv_dict: Dict[str, Any]) -> Dict[str, ConfigOption]:
        cv_map = {}
        method = ""
        for cv_name in cv_dict.keys():
            try:
                value = cv_dict[cv_name]
                if cv_name == Method.name:
                    method = value
                cv_object: ConfigOption = cross_validation_dict[cv_name](protocol=protocol, value=value)
                cv_object.transform_value_if_necessary()
                cv_map[cv_name] = cv_object
            except KeyError:
                raise ConfigurationException(f"Unknown cross validation option: {cv_name}!")

        if method == "":
            raise ConfigurationException(f"Required option method is missing from cross_validation_config!")
        else:
            # Add default value for choose by
            if ChooseBy.name not in cv_dict.keys():
                cv_map[ChooseBy.name] = ChooseBy(protocol=protocol)

        return cv_map

    @staticmethod
    def _get_config_maps(protocol: Protocol, config_dict: Dict[str, Any], config_file_path: Path = None) \
            -> Tuple[Dict[str, ConfigOption], Dict[str, CrossValidationOption]]:

        config_map = {}
        cv_map = {}
        contains_cross_validation_config = False
        for config_name in config_dict.keys():
            try:
                if config_name == CROSS_VALIDATION_CONFIG_KEY:
                    cv_map = Configurator._get_cross_validation_map(protocol=protocol,
                                                                    cv_dict=config_dict[config_name])
                    contains_cross_validation_config = True
                else:
                    value = config_dict[config_name]
                    if value == "":  # Ignore empty values
                        continue
                    config_object: ConfigOption = all_options_dict[config_name](protocol=protocol, value=value)
                    config_object.transform_value_if_necessary(config_file_path)
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
                config_object.transform_value_if_necessary(config_file_path)
                config_map[option.name] = config_object

        # Add default cross validation method if necessary
        if not contains_cross_validation_config:
            # hold_out by default, does not need any additional parameters
            cv_map[Method.name] = Method(protocol=protocol)
            cv_map[ChooseBy.name] = ChooseBy(protocol=protocol)

        return config_map, cv_map

    @staticmethod
    def _verify_config(protocol: Protocol, config_map: Dict[str, ConfigOption]):
        config_objects = list([config_object for config_object in config_map.values()])

        # Check rules
        all_rules = protocol_rules + config_option_rules
        for rule in all_rules:
            success, reason = rule.apply(protocol=protocol,
                                         config=config_objects)
            if not success:
                raise ConfigurationException(reason)

        # Check protocol and value
        for config_object in config_objects:
            if protocol not in config_object.allowed_protocols:
                raise ConfigurationException(f"{config_object.name} not allowed for protocol {protocol}!")

            if not config_object.check_value():
                raise ConfigurationException(f"{config_object.value} not valid for option {config_object.name}!")

    @staticmethod
    def _verify_cv_config(protocol: Protocol, config_map: Dict[str, ConfigOption],
                          cv_config: Dict[str, CrossValidationOption]):
        cv_objects = list([cv_object for cv_object in cv_config.values()])
        config_objects = list([config_object for config_object in config_map.values()])

        if Method.name not in cv_config.keys():
            raise ConfigurationException(f"Required option method is missing from cross_validation_config!")
        method = cv_config[Method.name]

        # Check rules
        for rule in cross_validation_rules:
            success, reason = rule.apply(protocol=protocol,
                                         config=cv_objects)
            if not success:
                raise ConfigurationException(reason)
        for rule in optimization_rules:
            success, reason = rule.apply(protocol, config=cv_objects + config_objects)

            if not success:
                raise ConfigurationException(reason)

        # Check method and value
        if method == "hold_out" and len(cv_objects) > 1:
            raise ConfigurationException(f"Cross validation method hold_out does not allow any other options!")

        for cv_object in cv_objects:
            if cv_object.cv_methods != [] and method.value not in cv_object.cv_methods:
                raise ConfigurationException(
                    f"Option {cv_object.name} not allowed for cross validation method {method.name}!")

            if not cv_object.check_value():
                raise ConfigurationException(
                    f"{cv_object.value} not valid for cross validation option {cv_object.name}!")

    def get_verified_config(self) -> Dict[str, Any]:
        config_map, cv_map = self._get_config_maps(protocol=self.protocol, config_dict=self._config_dict,
                                                   config_file_path=self._config_file_path)
        self._verify_config(protocol=self.protocol, config_map=config_map)
        self._verify_cv_config(protocol=self.protocol, config_map=config_map, cv_config=cv_map)
        result = {}
        for config_object in config_map.values():
            result[config_object.name] = config_object.value
        result[CROSS_VALIDATION_CONFIG_KEY] = {}
        for cv_object in cv_map.values():
            result[CROSS_VALIDATION_CONFIG_KEY][cv_object.name] = cv_object.value

        return result
