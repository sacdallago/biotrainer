import argparse

from .executer import parse_config_file_and_execute_run


def headless_main(config_file_path: str):
    parse_config_file_and_execute_run(config_file_path=config_file_path)


def main(args=None):
    """
    Pipeline commandline entry point
    """
    parser = argparse.ArgumentParser(description='Trains models on protein embeddings.')
    parser.add_argument('config_path', metavar='/path/to/pipeline_definition.yml', type=str, nargs='?',
                        help='The path to the config. For examples, see folder "examples".')

    arguments = parser.parse_args()

    config_path = arguments.config_path
    if not config_path:
        parser.print_help()
    else:
        parse_config_file_and_execute_run(config_path)


if __name__ == '__main__':
    main()
