from ...protocols import Protocol

# Unused dataset splits are commented out

FLIP_DATASETS = {
    "aav": {
        "splits": [
            # "des_mut",
            # "mut_des",
            # "one_vs_many",
            "two_vs_many",
            # "seven_vs_many",
            "low_vs_high",
            # "sampled"
        ],
        "evaluation_metric": "spearmans-corr-coeff",
        "protocol": Protocol.sequence_to_value,
    },
    "bind": {
        "splits": [
            # "one_vs_many",
            # "two_vs_many",
            "from_publication",
            # "one_vs_sm",
            # "one_vs_mn",
            # "one_vs_sn"
        ],
        "evaluation_metric": "macro-f1_score",
        "protocol": Protocol.residue_to_class,
    },
    # "conservation": {
    #     "splits": [
    #         "sampled"
    #     ],
    #     "evaluation_metric": "accuracy",
    #     "protocol": Protocol.residue_to_class,
    # },
    "gb1": {
        "splits": [
            # "one_vs_rest",
            "two_vs_rest",
            # "three_vs_rest",
            "low_vs_high",
            # "sampled"
        ],
        "evaluation_metric": "spearmans-corr-coeff",
        "protocol": Protocol.sequence_to_value,
    },
    "meltome": {
        "splits": [
            "mixed_split",
            # "human",
            # "human_cell"
        ],
        "evaluation_metric": "spearmans-corr-coeff",
        "protocol": Protocol.sequence_to_value,
    },
    "scl": {
        "splits": [
            # "mixed_soft",
            "mixed_hard",
            # "human_soft",
            # "human_hard",
            # "balanced",
            # "mixed_vs_human_2"
        ],
        "evaluation_metric": "accuracy",
        "protocol": Protocol.sequence_to_class,
    },
    # "sav": {
    #     "splits": [
    #         "mixed",
    #         # "human",
    #         # "only_savs"
    #     ],
    #     "evaluation_metric": "f1_score",
    #     "protocol": Protocol.sequence_to_class,
    # },
    "secondary_structure": {
        "splits": [
            "sampled"
        ],
        "evaluation_metric": "accuracy",
        "protocol": Protocol.residue_to_class,
    },
}
