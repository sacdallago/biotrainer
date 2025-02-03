import logging

def get_logger(name: str) -> logging.Logger:
    """Get a logger in the biotrainer namespace."""
    return logging.getLogger(f'biotrainer.{name}')