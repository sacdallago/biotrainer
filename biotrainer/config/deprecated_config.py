from typing import Dict, Any


def deprecated_config_keys(config_dict: Dict[str, Any]) -> Dict[str, str]:
    deprecated_keys_with_replacement = {"sequence_file": "input_file",
                                        "labels_file": "input_file",
                                        "mask_file": "input_file",
                                        }

    contained_deprecated_keys = {}
    for key in config_dict.keys():
        if key in deprecated_keys_with_replacement:
            contained_deprecated_keys[key] = deprecated_keys_with_replacement[key]
    return contained_deprecated_keys