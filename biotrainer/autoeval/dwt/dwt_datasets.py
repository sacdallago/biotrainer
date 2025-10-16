from ...protocols import Protocol

# Unused dataset splits are commented out

DWT_DATASETS = {
    "scl": {
        "evaluation_metric": "accuracy",
        "protocol": Protocol.sequence_to_class,
    },
    "secondary_structure": {
        "evaluation_metric": "accuracy",
        "protocol": Protocol.residue_to_class,
    },
}
