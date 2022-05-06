# Pytest configuration file
# Creates a test for each of the combination of protocol, model, embedder_name

from biotrainer.models import __MODELS

protocols = ["residue_to_class", "sequence_to_class"]
models = ["FNN", "CNN", "LogReg"]
embedder_names = ["one_hot_encoding", "word2vec"]

all_params = {
    "test_config": {
        "params": ["protocol", "model", "embedder_name", "should_fail"],
        "values": []
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
