from ...protocols import Protocol

PBC_DATASETS = {
    "conservation": {
        "evaluation_metric": "accuracy",
        "protocol": Protocol.residue_to_class,
    },
    "disorder": {
        "evaluation_metric": "spearmans-corr-coeff",
        "protocol": Protocol.residue_to_value,
    },
    "scl": {
        "evaluation_metric": "accuracy",
        "protocol": Protocol.sequence_to_class,
    },
    "secondary_structure": {
        "evaluation_metric": "accuracy",
        "protocol": Protocol.residue_to_class,
    },
}
