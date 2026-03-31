import os

from pathlib import Path
from typing import List, Optional

from ..core import AutoEvalDataHandler, AutoEvalTask


def setup_output_dir(base_dir: Path, embedder_name: str, framework_name: str) -> Path:
    embedder_dir_name = embedder_name
    if "/" in embedder_dir_name:  # Huggingface
        embedder_dir_name = embedder_dir_name.replace("/", "-")

    output_dir = Path(base_dir) / embedder_dir_name / framework_name

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    return output_dir


def setup_pipeline(data_handler: AutoEvalDataHandler,
                   min_seq_length: Optional[int] = None,
                   max_seq_length: Optional[int] = None,
                   custom_storage_path: Optional[str] = None,
                   force_download: Optional[bool] = False,
                   ) -> List[AutoEvalTask]:
    framework_base_path = data_handler.get_framework_base_path(
        custom_storage_path=custom_storage_path)

    if force_download:
        data_handler.clear_autoeval_cache()

    if not os.path.exists(framework_base_path):
        os.makedirs(framework_base_path, exist_ok=True)

    if data_handler.is_download_necessary(framework_base_path):
        data_handler.download_data(data_dir=framework_base_path)
    data_handler.preprocess(base_path=framework_base_path,
                            min_seq_length=min_seq_length,
                            max_seq_length=max_seq_length)
    auto_eval_tasks = data_handler.get_tasks(base_path=framework_base_path,
                                             min_seq_length=min_seq_length,
                                             max_seq_length=max_seq_length)

    return auto_eval_tasks
