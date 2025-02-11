import os
import torch
import logging

from typing import Optional
from tqdm import tqdm as tqdm_console
from tqdm.notebook import tqdm as tqdm_notebook


def _is_running_in_notebook() -> bool:
    try:
        # This will only exist in Jupyter environments
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':  # Jupyter notebook or qtconsole
            return True
        elif shell == 'TerminalInteractiveShell':  # IPython terminal
            return False
    except NameError:
        return False
    return False


class _TqdmLoggingHandler(logging.Handler):
    def __init__(self, tqdm_instance: Optional[tqdm_notebook] = None):
        super().__init__()
        self.tqdm_instance = tqdm_instance
        self.current_epoch = 0

    def emit(self, record):
        try:
            update_total = False
            if "Epoch" in record.msg:
                self.current_epoch = int(record.msg.split("Epoch ")[-1].strip())
            if "Running final evaluation on the best model" in record.msg:
                update_total = True
            if self.tqdm_instance:
                if len(record.msg) > 40:
                    record.msg = record.msg[:37] + "..."
                msg = self.format(record)
                self.tqdm_instance.set_description(msg)
                self.tqdm_instance.n = self.current_epoch
                if update_total:
                    self.tqdm_instance.total = self.current_epoch
                self.tqdm_instance.refresh()
            else:
                msg = self.format(record)
                tqdm_console.write(msg)
        except Exception:
            self.handleError(record)


def get_logger(name: str) -> logging.Logger:
    """Get a logger in the biotrainer namespace."""
    return logging.getLogger(f'biotrainer.{name}')


def setup_logging(output_dir: str, num_epochs: int):
    # Disable logging during test execution because of problems in Windows
    if "PYTEST_CURRENT_TEST" in os.environ:
        return

    logging.captureWarnings(True)
    biotrainer_logger = logging.getLogger('biotrainer')
    biotrainer_logger.propagate = False  # Prevent propagation to root logger

    # Set up handlers
    file_handler = logging.FileHandler(output_dir + "/logger_out.log")

    # Different handling for notebook vs console
    is_notebook = _is_running_in_notebook()
    if is_notebook:
        # Create a tqdm progress bar for notebooks
        progress_bar = tqdm_notebook(total=num_epochs)
        stream_handler = _TqdmLoggingHandler(progress_bar)
    else:
        stream_handler = logging.StreamHandler()  # Regular console output

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    biotrainer_logger.setLevel(logging.INFO)
    file_handler.setLevel(logging.INFO)
    stream_handler.setLevel(logging.INFO)

    biotrainer_logger.addHandler(file_handler)
    biotrainer_logger.addHandler(stream_handler)

    # Only log errors for onnx and dynamo
    torch._logging.set_logs(dynamo=logging.ERROR, onnx=logging.ERROR, onnx_diagnostics=False)
    for logger_name in ["torch.onnx", "torch._dynamo", "onnxscript"]:
        logging.getLogger(logger_name).setLevel(logging.ERROR)


def clear_logging():
    biotrainer_logger = logging.getLogger('biotrainer')

    for handler in biotrainer_logger.handlers:
        biotrainer_logger.removeHandler(handler)
        handler.close()
