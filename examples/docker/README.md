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