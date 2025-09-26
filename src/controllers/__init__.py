REGISTRY = {}

from .basic_controller import BasicMAC
from .non_shared_controller import NonSharedMAC
from .maddpg_controller import MADDPGMAC
from .ExpoComm_controller import ExpoCommMAC
from .DRMAC_controller import DRMACMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["non_shared_mac"] = NonSharedMAC
REGISTRY["maddpg_mac"] = MADDPGMAC
REGISTRY["ExpoComm_mac"] = ExpoCommMAC
REGISTRY["DRMAC_mac"] = DRMACMAC