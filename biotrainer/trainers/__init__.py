from .ResidueTrainer import ResidueTrainer
from .SequenceTrainer import SequenceTrainer

__TRAINERS = {
    'residue_to_class': ResidueTrainer,
    'sequence_to_class': SequenceTrainer
}


def get_trainer(**kwargs):
    protocol = kwargs['protocol']
    trainer = __TRAINERS.get(protocol)

    if not trainer:
        raise NotImplementedError
    else:
        return trainer.pipeline(**kwargs)


__all__ = [
    'get_trainer'
]
