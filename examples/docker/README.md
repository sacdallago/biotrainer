# docker example

This example shows how to use a *biotrainer* docker container to execute training runs. It uses the residue_to_class
protocol. The container works with all existing protocols.
Sequence and labels files get mounted before the training starts. Output can be found in this directory.

```bash
# Go to the base directory
cd /home/user/biotrainer/  
# Build
docker build -t biotrainer .
# Run
docker run --rm \
    -v "$(pwd)/examples/docker":/mnt \
    -u $(id -u ${USER}):$(id -g ${USER}) \
    biotrainer:latest /mnt/config.yml
```

Note that the image created by the standard `Dockerfile` does no longer include 
[bio_embeddings](https://github.com/sacdallago/bio_embeddings/).
If you do want to use `bio_embeddings` (only possible with biotrainer version <= 0.7.0), 
you can simply change the installation command in the Dockerfile:
```dockerfile
RUN python3 -m venv .venv && \
    # Install a recent version of pip, otherwise the installation of many linux2010 packages will fail
    .venv/bin/pip install -U pip && \
    # Make sure poetry install the metadata for biotrainer
    mkdir biotrainer && \
    touch biotrainer/__init__.py && \
    touch README.md && \
    $HOME/.local/bin/poetry config virtualenvs.in-project true && \
    # CHANGE THIS LINE:
    $HOME/.local/bin/poetry install --no-dev --all-extras 
```