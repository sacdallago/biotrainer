from enum import Enum
from typing import Optional, Union

from .core import AutoEvalFramework
from .frameworks import PBCSupervisedFramework, FLIPFramework, PGYMFramework, PBCZeroShotContactFramework, \
    PBCSupervisedContactFramework, PBCUnsupervisedFramework


class AvailableFramework(Enum):
    FLIP = "FLIP"
    PBC_SUPERVISED = "PBC_SUPERVISED"
    PBC_UNSUPERVISED = "PBC_UNSUPERVISED"
    PGYM = "PGYM"
    PBC_ZEROSHOT_CONTACT = "PBC_ZEROSHOT_CONTACT"
    PBC_SUPERVISED_CONTACT = "PBC_SUPERVISED_CONTACT"

    @staticmethod
    def all():
        return [AvailableFramework.PBC_SUPERVISED,
                AvailableFramework.PBC_UNSUPERVISED,
                AvailableFramework.PGYM,
                AvailableFramework.PBC_ZEROSHOT_CONTACT,
                AvailableFramework.PBC_SUPERVISED_CONTACT,
                AvailableFramework.FLIP]

    @staticmethod
    def dashboard_frameworks():
        return [AvailableFramework.PBC_SUPERVISED,
                AvailableFramework.PGYM,
                AvailableFramework.PBC_ZEROSHOT_CONTACT,
                AvailableFramework.PBC_SUPERVISED_CONTACT,
                ]


available_frameworks = {AvailableFramework.FLIP: FLIPFramework(),
                        AvailableFramework.PBC_SUPERVISED: PBCSupervisedFramework(),
                        AvailableFramework.PBC_UNSUPERVISED: PBCUnsupervisedFramework(),
                        AvailableFramework.PGYM: PGYMFramework(),
                        AvailableFramework.PBC_ZEROSHOT_CONTACT: PBCZeroShotContactFramework(),
                        AvailableFramework.PBC_SUPERVISED_CONTACT: PBCSupervisedContactFramework(),
                        }


def framework_factory(framework_name: Union[str, AvailableFramework]) -> Optional[AutoEvalFramework]:
    try:
        av_framework = AvailableFramework(framework_name.upper()) if isinstance(framework_name, str) else framework_name
        return available_frameworks.get(av_framework, None)
    except ValueError:
        return None
