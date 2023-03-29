import argparse

from .executer import parse_config_file_and_execute_run


def headless_main(config_file_path: str):
    parse_config_file_and_execute_run(config_file_path=config_file_path)


def _list_available_embedders():
    from bio_embeddings.embed import __all__

    for imported_embedder in __all__:
        if "Interface" not in imported_embedder:
            print(imported_embedder)


def main(args=None):
    """
    Pipeline commandline entry point
    """
    parser = argparse.ArgumentParser(description='Trains models on protein embeddings.')
    parser.add_argument('--list_embedders', required=False, action='store_true',
                        help='List all available embedders from bio_embeddings')
    parser.add_argument('config_path', metavar='/path/to/pipeline_definition.yml', type=str, nargs='?',
                        help='The path to the config. For examples, see folder "examples".')

    arguments = parser.parse_args()

    if arguments.list_embedders:
        _list_available_embedders()
        return

    config_path = arguments.config_path
    if not config_path:
        parser.print_help()
    else:
        parse_config_file_and_execute_run(config_path)


if __name__ == '__main__':
    main()
