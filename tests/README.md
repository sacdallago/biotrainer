# Biotrainer Tests

## Configuration tests

Tests are run with pytest. 
`conftest.py` will take care that a test is created for all available protocols, models and 
two embedding methods (one_hot_encoding, word2vec).
`def test_config(protocol, model, embedder_name):` executes them afterwards and checks, that an `out.yml` file is 
created (hence the pipeline worked).

```bash
# cd biotrainer/tests
pytest
# Disable warnings:
pytest --disable-warnings
# To show output on passed tests:
pytest --disable-warnings -rP
```