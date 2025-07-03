import os
import torch
import logging

from .execution_environment import is_running_in_notebook

class _NotebookLogHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        try:
            from IPython.display import display, HTML
            self.display = display
            self.HTML = HTML
            self.display("", clear=True)
            self.display(self.HTML('''
                <div id="biotrainer_status" style="margin: 10px 0;">
                    Biotrainer Training Running...
                </div>
            '''))
        except ImportError as e:
            raise ImportError("You are running in a notebook environment, but missing required dependencies. "
                            "Please install them via `uv pip install -e '.[jupyter]'") from e

    def emit(self, record):
        self.display(self.HTML(f'''
            <script>
                document.getElementById("biotrainer_status").innerHTML = "{self.format(record)}";
            </script>
        '''))

    def close(self):
        self.display(self.HTML('''
            <script>
                document.getElementById("biotrainer_status").innerHTML = "Biotrainer Training Finished!";
            </script>
        '''))
        super().close()


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

    # Remove existing handlers if they are already there
    existing_handlers = biotrainer_logger.handlers
    for handler in existing_handlers:
        biotrainer_logger.removeHandler(handler)

    # Set up handlers
    file_handler = logging.FileHandler(output_dir + "/logger_out.log")

    # Different handling for notebook vs console
    is_notebook = is_running_in_notebook()
    if is_notebook:
        # Create a tqdm progress bar like handler for notebooks instead of plain text output
        stream_handler = _NotebookLogHandler()
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
