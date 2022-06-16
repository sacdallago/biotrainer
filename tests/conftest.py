# Pytest configuration file
# Creates a test for each of the combination of protocol, model, embedder_name

from biotrainer.models import __MODELS
from biotrainer.utilities.executer import __PROTOCOLS


def get_all_available_models():
    all_models_by_protocol = [list(protocol.keys()) for protocol in __MODELS.values()]
    all_models_list = [model for model_list in all_models_by_protocol for model in model_list]
    return set(all_models_list)


protocols = __PROTOCOLS
models = get_all_available_models()
embedder_names = ["one_hot_encoding", "word2vec"]

all_params = {
    'test_config': {
        'params': ["protocol", "model", "embedder_name", "should_fail"],
        'values': []
    },
}


def pytest_generate_tests(metafunc):
    for protocol in protocols:
        for model in models:
            should_fail = True if model not in __MODELS.get(protocol) else False
            for embedder_name in embedder_names:
                all_params["test_config"]["values"].append([protocol, model, embedder_name, should_fail])
    fct_name = metafunc.function.__name__
    if fct_name in all_params:
        params = all_params[fct_name]
        metafunc.parametrize(params["params"], params["values"])
