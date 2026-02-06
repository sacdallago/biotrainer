from enum import Enum
from typing import Optional, Union

from .pbc import PBCFramework
from .flip import FLIPFramework
from .pgym import PGYMFramework
from .core import AutoEvalFramework


class AvailableFramework(Enum):
    FLIP = "FLIP"
    PBC = "PBC"
    PGYM = "PGYM"


available_frameworks = {AvailableFramework.FLIP: FLIPFramework(),
                        AvailableFramework.PBC: PBCFramework(),
                        AvailableFramework.PGYM: PGYMFramework()
                        }


def framework_factory(framework_name: Union[str, AvailableFramework]) -> Optional[AutoEvalFramework]:
    try:
        av_framework = AvailableFramework(framework_name.upper()) if isinstance(framework_name, str) else framework_name
        return available_frameworks.get(av_framework, None)
    except ValueError:
        return None