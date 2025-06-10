import json
import hashlib

from pathlib import Path
from typing import Dict, Any, List


def calculate_sequence_hash(sequence: str) -> str:
    suffix = len(sequence)
    sequence = f"{sequence}_{suffix}"
    return hashlib.sha256(sequence.encode()).hexdigest()


def calculate_model_hash(
        dataset_files: List[Path],
        config: Dict[Any, Any],
        custom_trainer: bool,
) -> str:
    """
    Create a deterministic hash representing dataset files and model configuration.

    Args:
        dataset_files: List of paths to dataset files
        config: Dictionary containing model configuration
        custom_trainer: If true, a custom trainer is used

    Returns:
        A hex string hash uniquely identifying this model setup
    """
    # 1. Calculate file hashes
    file_hashes = []
    for file_path in sorted(dataset_files):  # Sort for deterministic order
        if file_path.exists():
            file_hash = hashlib.sha256()
            with open(file_path, 'rb') as f:
                # Read in chunks to handle large files
                for chunk in iter(lambda: f.read(4096), b''):
                    file_hash.update(chunk)
            file_hashes.append((file_path.name, file_hash.hexdigest()))
        else:
            raise FileNotFoundError(f"Did not find {file_path} during model hashing!")

    # 2. Prepare config hash (normalize it to ensure consistency)
    # Sort keys and convert everything to string
    config_normalized = json.dumps({str(k): str(v) for k, v in config.items()}, sort_keys=True)
    config_hash = hashlib.sha256(config_normalized.encode()).hexdigest()

    # 3. Combine all hashes
    combined = {
        'files': file_hashes,
        'config': config_hash,
        'custom_trainer': str(custom_trainer)
    }

    # 4. Create final hash
    final_hash = hashlib.sha256(json.dumps(combined, sort_keys=True).encode()).hexdigest()

    return final_hash[:16]  # First 16 chars is sufficient for us