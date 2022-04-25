#!/usr/bin/env python3

import argparse
import logging

from biotrainer.utilities.executer import parse_config_file_and_execute_run


def main():
    """
    Pipeline commandline entry point
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    # Jax likes to print warnings
    logging.captureWarnings(True)

    parser = argparse.ArgumentParser(description='Trains models on protein embeddings.')
    parser.add_argument('config_path', metavar='/path/to/pipeline_definition.yml', type=str, nargs=1,
                        help='The path to the config. For examples, see folder "parameter examples".')
    arguments = parser.parse_args()

    parse_config_file_and_execute_run(arguments.config_path[0])


if __name__ == '__main__':
    main()