
from .protocols import Protocols
from .config_rules import MutualExclusive, ProtocolRequires, ProtocolProhibits
from .input_options import SequenceFile, LabelsFile, MaskFile
from .training_options import AutoResume, PretrainedModel
from .embedding_options import EmbedderName, EmbeddingsFile

# Optional attribute for ConfigOption!

protocol_rules = [
    ProtocolRequires(protocol=Protocols.per_residue_protocols(), requires=[SequenceFile, LabelsFile]),
    ProtocolRequires(protocol=Protocols.per_protein_protocols(), requires=[SequenceFile]),
    ProtocolProhibits(protocol=Protocols.per_protein_protocols(), prohibits=[LabelsFile, MaskFile])
]

config_option_rules = [
    MutualExclusive(exclusive=[AutoResume, PretrainedModel]),
    MutualExclusive(exclusive=[EmbedderName, EmbeddingsFile])
]
