# docker example

This example shows how to use a *biotrainer* docker container to execute training runs. It uses the residue_to_class
protocol, but docker, of course, works with all existing protocols.
Sequence and labels files get mounted before the training starts. Output can be found afterwards in this directory.

```bash
# Go to the base directory
cd /home/user/biotrainer/  
# Build
docker build -t biotrainer .
# Run
docker run --rm \
    -v "$(pwd)/examples/docker":/mnt \
    -v bio_embeddings_weights_cache:/root/.cache/bio_embeddings \
    -u $(id -u ${USER}):$(id -g ${USER}) \
    biotrainer:latest /mnt/config.yml
```