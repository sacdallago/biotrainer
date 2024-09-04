# Biotrainer Troubleshooting

## Installation

**[WINDOWS] Cannot install via `poetry install` - `setuptools/numpy/networkx` cannot be installed**

-> Try to enable [long paths in the registry](https://learn.microsoft.com/en-us/windows/win32/fileio/maximum-file-path-limitation?tabs=registry#enable-long-paths-in-windows-10-version-1607-and-later)

## Running

**[WINDOWS] Default IDE run configuration does not work**

-> Only use script name as executable: `run-biotrainer.py` 
-> Put path into script parameters, e.g.: `examples/sequence_to_class/config.yml

**[WINDOWS] PyTorch compile does not work because `triton` is missing**:

-> Use `disable_pytorch_compile: True` as a workaround for now