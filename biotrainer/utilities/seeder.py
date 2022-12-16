import logging
import torch
import random

import numpy as np

logger = logging.getLogger(__name__)


# https://discuss.pytorch.org/t/reproducibility-with-all-the-bells-and-whistles/81097
def seed_all(seed=42):
    logger.info(f"Using Seed: {seed}")

    random.seed(seed)

    np.random.seed(seed)  # Also seeds sklearn

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False