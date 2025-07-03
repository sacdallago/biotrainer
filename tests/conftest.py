# Pytest configuration file

from copy import deepcopy
from itertools import product
from biotrainer.protocols import Protocol
from biotrainer.models import get_available_models_dict, get_available_models_set

protocols = set([protocol.name for protocol in Protocol.all()])
models = get_available_models_set()
embedder_names = ["one_hot_encoding"]

all_params = {
    'test_protocol_config': {
        'params': ["protocol", "model", "embedder_name", "should_fail"],
        'values': [],
    },
    'test_cross_validation': {
        'params': ["cv_config"],
        'values': []
    }
}


def generate_tests_protocols(metafunc):
    protocols.add("residue_to_class-error1")
    protocols.add("residue_to_class-error2")
    for protocol in protocols:
        if "error" in protocol:
            # Test erroneous config only with one model and embedder, should_fail=True
            all_params["test_protocol_config"]["values"].append([protocol, "CNN", "one_hot_encoding", True])
            continue
        for model in models:
            should_fail = True if model not in get_available_models_dict().get(
                Protocol.from_string(protocol)) else False
            for embedder_name in embedder_names:
                all_params["test_protocol_config"]["values"].append([protocol, model, embedder_name, should_fail])

    fct_name = metafunc.function.__name__
    if fct_name in all_params:
        params = all_params[fct_name]
        metafunc.parametrize(params["params"], params["values"])


def generate_tests_cross_validation(metafunc):
    cv_methods_params = {
        "hold_out": {},
        "k_fold": {
            "k": [3, 3],
            "repeat": [1, 2],
            "stratified": [True, False],
            "nested": [True, False],
            "nested_k": [3, 3],
            "search_method": ["random_search", "grid_search"],
            "n_max_evaluations_random": [2, 2]
        },
        "leave_p_out": {
            "p": 1
        }
    }
    idx = 0
    for method, config in cv_methods_params.items():
        method_configuration = {
            "method": method
        }
        # Constant params
        for key, value in config.items():
            if type(value) is not list:
                method_configuration[key] = value
        # k_fold parameter permutations
        if method == "k_fold":
            unique_permutations = set([frozenset(dict(zip(config, v)).items()) for v in product(*config.values())])
            for unique_permutation in unique_permutations:
                unique_permutation_dict = dict(unique_permutation)
                if unique_permutation_dict["repeat"] > 1 and unique_permutation_dict["nested"] is True:
                    continue

                tmp_config = deepcopy(method_configuration)
                tmp_config.update(unique_permutation_dict)
                all_params["test_cross_validation"]["values"].append([tmp_config])
        else:
            all_params["test_cross_validation"]["values"].append([method_configuration])
        idx += 1

    fct_name = metafunc.function.__name__
    if fct_name in all_params:
        params = all_params[fct_name]
        metafunc.parametrize(params["params"], params["values"])


def pytest_generate_tests(metafunc):
    fct_name = metafunc.function.__name__
    if fct_name == "test_protocol_config":
        generate_tests_protocols(metafunc)
    elif fct_name == "test_cross_validation":
        generate_tests_cross_validation(metafunc)
