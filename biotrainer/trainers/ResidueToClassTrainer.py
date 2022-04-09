import itertools
import logging
import torch

import numpy as np

from collections import Counter
from typing import Union, List, Dict, Tuple, Set


from ..utilities import get_device, read_FASTA, attributes_from_seqrecords

logger = logging.getLogger(__name__)


class ResidueToClassTrainer:
