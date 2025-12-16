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
    "membrane": {
        "evaluation_metric": "macro-f1_score",
        "protocol": Protocol.residue_to_class,
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
