import random
import itertools

from typing import Dict, Any, Generator, Union

from ..utilities import get_logger
from ..config import ConfigurationException

logger = get_logger(__name__)


class HyperParameterManager:
    def __init__(self, **kwargs):
        self._constant_params = {}
        self._params_to_optimize = {}
        for key, value in kwargs.items():
            if key == "input_data":
                continue
            if "range" in str(value):
                try:
                    compiled_range = compile(source=value, filename='<range>', mode='eval')
                    range_obj = eval(compiled_range, locals())
                    assert range_obj, "Range could not be compiled from input!"
                    self._params_to_optimize[key] = range_obj
                except Exception as e:
                    raise ConfigurationException(f"Unable to compile hyper_parameters from config:\n"
                                                 f"{key}: {value}") from e
            elif type(value) is list:
                self._params_to_optimize[key] = list(set(value))
            elif type(value) is str and "[" in value and "]" in value:
                try:
                    compiled_list = compile(source=value, filename='<list>', mode='eval')
                    list_obj = eval(compiled_list, locals())
                    assert list_obj, "List could not be compiled from input!"
                    self._params_to_optimize[key] = list_obj
                except Exception as e:
                    raise ConfigurationException(f"Unable to compile hyper_parameters from config:\n"
                                                 f"{key}: {value}") from e
            else:
                self._constant_params[key] = value

        if "nested" in self._constant_params["cross_validation_config"].keys() and \
                str(self._constant_params["cross_validation_config"]["nested"]).capitalize() == "True":
            if len(self._params_to_optimize.keys()) == 0:
                raise ConfigurationException(f"No parameters to optimize were given. Define with list or range!\n"
                                             f"e.g. learning_rate: [1e-3, 1e-4]")
        if len(self._params_to_optimize.keys()) > 0 and \
                self._constant_params["cross_validation_config"]["method"] != "k_fold":
            raise ConfigurationException(f"Parameter search only supported for nested k_fold cross validation!")

    def search(self, mode="no_search") -> \
            Generator[Dict[str, Any], None, Union[None, Generator[Dict[str, Any], None, None]]]:
        if mode == "no_search":
            return self._no_search()
        elif mode == "grid_search":
            return self._grid_search()
        elif mode == "random_search":
            return self._random_search()
        else:
            raise NotImplementedError(f"Hyper parameter search method {mode} not implemented!")

    def get_only_params_to_optimize(self, hyper_params: Dict[str, Any]) -> Dict[str, Any]:
        hp_to_optimize = {}
        for key, value in hyper_params.items():
            if key in self._params_to_optimize.keys():
                hp_to_optimize[key] = value
        return hp_to_optimize

    def _no_search(self) -> Generator[Dict[str, Any], None, None]:
        yield self._constant_params

    def _grid_search(self) -> Generator[Dict[str, Any], None, None]:
        keys, values = zip(*self._params_to_optimize.items())

        for combination in itertools.product(*values):
            hyper_parameters = dict(zip(keys, combination))
            yield {**self._constant_params, **hyper_parameters}

    def _random_search(self) -> Generator[Dict[str, Any], None, None]:
        n_max_evaluations = self._constant_params["cross_validation_config"]["n_max_evaluations_random"]
        all_possible_hp_combinations = list(self._grid_search())
        if n_max_evaluations >= len(all_possible_hp_combinations):
            logger.warning(f"Number of max_evaluations ({n_max_evaluations}) >= "
                           f"Number of possible combinations ({len(all_possible_hp_combinations)}) for random search "
                           f"hyperparameter optimization! Performing grid search instead..")
            yield from self._grid_search()

        random.shuffle(all_possible_hp_combinations)

        for i in range(n_max_evaluations):
            yield {**self._constant_params, **all_possible_hp_combinations[i]}
